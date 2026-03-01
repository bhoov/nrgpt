"""
Inference and evaluation script for NRGPT models.

Loads pre-trained checkpoints, runs MMLU benchmark evaluation via deepeval,
generates text samples, computes quality metrics, and logs everything to wandb.

Usage (single GPU):
    uv run python inference.py --model_path=models.NRGPT_H_FF2W

Usage (with config file):
    uv run python inference.py config/train_owt_config.py --model_path=models.NRGPT_H_FF2W
"""
#%%
import os
import time
import math
import pickle
import inspect
import re
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

import tyro
import tiktoken

import attrs
from cattrs import unstructure, structure
from typing import *
from pathlib import Path
from utils import is_interactive
import json
import yaml
import sys
from model_config import ModelConfig
import importlib
from evaluate_nrgpt import *
from torch.utils.flop_counter import FlopCounterMode

from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.models import DeepEvalBaseLLM

print('check', torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)

# =============================================================================
# DeepEval model wrapper for MMLU benchmarking
# =============================================================================

class GPTDeepEvalWrapper(DeepEvalBaseLLM):
    """Wraps a GPT model to be compatible with the deepeval benchmarking interface.

    Handles tiktoken encoding/decoding, greedy generation, prompt stripping,
    and answer extraction (A/B/C/D) for multiple-choice benchmarks like MMLU.
    """

    def __init__(self, model, enc, device='cuda', max_new_tokens=50, debug=False):
        self.model = model
        self.enc = enc
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.debug = debug

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None) -> str:
        """Generate text from prompt. If schema is provided, extract a multiple-choice answer."""
        input_ids = torch.tensor(self.enc.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)

        if self.debug:
            print(f"Prompt length: {input_ids.shape[1]} tokens")
            print(f"First 100 chars of prompt: {prompt[:100]}...")

        with torch.no_grad():
            generated_ids = self.model.generate(
                idx=input_ids,
                max_new_tokens=self.max_new_tokens,
                greedy=True
            )

        # Decode only the newly generated tokens (model returns prompt + generated)
        generated_ids_list = generated_ids[0].cpu().tolist()
        filtered_ids = [t for t in generated_ids_list if t <= self.enc.max_token_value]
        full_text = self.enc.decode(filtered_ids)

        # Strip the prompt from the output
        if full_text.startswith(prompt):
            generated_text = full_text[len(prompt):].strip()
        else:
            # Fallback: strip by token count if string prefix doesn't match exactly
            prompt_tokens = self.enc.encode(prompt)
            new_tokens = filtered_ids[len(prompt_tokens):]
            generated_text = self.enc.decode(new_tokens).strip()

        if self.debug:
            print(f"Generated text: '{generated_text[:100]}...'")

        # For MMLU: extract A/B/C/D answer and return structured response
        if schema is not None:
            answer = self._extract_answer(generated_text)
            if callable(schema):
                try:
                    return schema(answer=answer)
                except Exception:
                    return answer
            else:
                return {"answer": answer}

        return generated_text

    def _extract_answer(self, text: str) -> str:
        """Extract a multiple-choice answer (A/B/C/D) from generated text.

        Uses a priority-ordered list of regex patterns, from most specific
        ("The answer is A") to least specific (any standalone A-D letter).
        Falls back to "A" if no valid answer is found.
        """
        text = text.strip().upper()

        if not text:
            print("Warning: Empty generated text")
            return "A"

        # Priority patterns (most specific to least specific)
        patterns = [
            r'(?:THE\s+)?ANSWER\s+IS\s*:?\s*([A-D])',
            r'(?:CORRECT\s+)?ANSWER\s*:?\s*([A-D])',
            r'CHOICE\s*:?\s*([A-D])',
            r'^([A-D])(?:[\.)\s]|$)',
            r'\(([A-D])\)',
            r'([A-D])[\.)\s]',
            r'([A-D])\s*$',
            r'\b([A-D])\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                answer = match.group(1)
                if self.debug:
                    print(f"Found answer '{answer}' with pattern: {pattern}")
                return answer

        if any(garbage in text for garbage in ['RUNES', 'OPS', 'VET', 'GEN', 'ATK', 'XII']):
            print(f"Warning: Garbage output detected: '{text[:50]}...'")
        else:
            print(f"Warning: No valid answer found in: '{text[:50]}...'")

        return "A"

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model.__class__.__name__

# =============================================================================
# Configuration
# =============================================================================

@tyro.conf.configure(tyro.conf.FlagConversionOff, tyro.conf.OmitArgPrefixes)
@attrs.define
class TrainingConfig:
    # I/O
    out_dir: str = 'out'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    init_from: str = 'scratch'

    # wandb logging
    wandb_log: bool = False
    wandb_project: str = 'owt'
    wandb_run_name: str = 'gpt2'
    wandb_entity: Optional[str] = None

    # optimizer (needed for model init from resume)
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # learning rate decay
    decay_lr: bool = True
    cosine_decay: bool = False
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: Optional[float] = None
    min_lr_divfactor: float = 10.

    # data
    dataset: str = 'openwebtext'
    gradient_accumulation_steps: int = 5*8
    batch_size: int = 12

    # DDP settings
    backend: str = 'nccl'

    # system
    device: str = 'cuda'
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile: bool = True

    # model
    model_config: ModelConfig = ModelConfig()
    model_path: str = "models.NRGPT_H_FF2W"

    def get_wandb_run_name(self) -> str:
        is_cos = "_cos" if self.cosine_decay else "_exp"
        base_params = f'_model={self.model_path.rsplit(".", 1)[-1]}_embed={self.model_config.n_embed}_depth={self.model_config.n_layer}_heads={self.model_config.n_head}_LR={self.learning_rate}_minLR={self.min_lr}_minLrDiv={self.min_lr_divfactor}_numIter={self.max_iters}{is_cos}'
        return f'{self.wandb_run_name}{base_params}'

    def get_min_lr(self) -> float:
        if self.min_lr is None:
            return self.learning_rate / self.min_lr_divfactor
        return self.min_lr

    @property
    def block_size(self) -> int: return self.model_config.block_size


def load_config_from_file(config_file: str) -> TrainingConfig:
    """Load config from .py, .json, or .yaml file and merge with base config."""
    path = Path(config_file)

    if path.suffix == '.py':
        config_globals = {}
        exec(path.read_text(), config_globals)
        overrides = {k: v for k, v in config_globals.items() if not k.startswith('_')}
    elif path.suffix in ['.json']: overrides = json.loads(path.read_text())
    elif path.suffix in ['.yaml', '.yml']: overrides = yaml.safe_load(path.read_text())
    else: raise ValueError(f"Unsupported config file format: {path.suffix}")

    model_config_fields = set(attrs.fields_dict(ModelConfig).keys())
    training_config_fields = set(attrs.fields_dict(TrainingConfig).keys())

    model_config_overrides = {k: v for k, v in overrides.items() if k in model_config_fields}
    training_config_overrides = {k: v for k, v in overrides.items() if k in training_config_fields}

    if model_config_overrides:
        base_model_config = unstructure(ModelConfig())
        base_model_config.update(model_config_overrides)
        training_config_overrides['model_config'] = base_model_config

    return structure(training_config_overrides, TrainingConfig)

# =============================================================================
# Parse configuration
# =============================================================================

if is_interactive():
    conf = TrainingConfig()
else:
    conf = TrainingConfig()

    config_file = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        config_file = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        print("config_file:", config_file)
        conf = load_config_from_file(config_file)

    conf = tyro.cli(TrainingConfig, default=conf)

conf: TrainingConfig = conf
config_dict = unstructure(conf)
print("config_dict:", config_dict)
print("conf:", conf)

#%%
wandb_run_name = conf.get_wandb_run_name()
print('wandb_run_name:', wandb_run_name)
print("dataset:", conf.dataset)

# =============================================================================
# DDP / device setup
# =============================================================================

ddp = int(os.environ.get('RANK', -1)) != -1
print('rank:', int(os.environ.get('RANK', -1)), 'ddp:', ddp)
if ddp:
    init_process_group(backend=conf.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert conf.gradient_accumulation_steps % ddp_world_size == 0
    conf.gradient_accumulation_steps //= ddp_world_size
else:
    device = conf.device
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = conf.gradient_accumulation_steps * ddp_world_size * conf.batch_size * conf.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(conf.out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[conf.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# =============================================================================
# Data loader (needed for vocab_size detection from meta.pkl)
# =============================================================================

data_dir = os.path.join('data', conf.dataset)

def get_batch(split):
    """Load a random batch from the memory-mapped dataset."""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    elif split == 'val':
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    else:
        raise ValueError("Please input either train or val")
    ix = torch.randint(len(data) - conf.block_size, (conf.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+conf.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+conf.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Derive vocab_size from dataset metadata if available
iter_num = 0
best_val_loss = 1e9
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# =============================================================================
# Model initialization
# =============================================================================

def load_model_from_path(model_path: str):
    """Load model class from a dotted path like 'models.NRGPT_H_FF2W'."""
    if '.' not in model_path:
        raise ValueError(f"model_path must be in 'module.ClassName' format, got: {model_path}")
    module_name, class_name = model_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

gptconf = conf.model_config
Model = load_model_from_path(conf.model_path)

if conf.init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None: print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    gptconf.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
    model = Model(gptconf)
elif conf.init_from == 'resume':
    print(f"Resuming from {conf.out_dir}")
    ckpt_path = os.path.join(conf.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    gptconf = attrs.evolve(gptconf, **checkpoint_model_args)
    model = Model(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif conf.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {conf.init_from}")
    override_args = dict(dropout=conf.model_config.dropout)
    model = Model.from_pretrained(conf.init_from, override_args)
    checkpoint_overrides = {}
    for k in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias', 'vocab_size']:
        checkpoint_overrides[k] = getattr(model.config, k)
    gptconf = attrs.evolve(gptconf, **checkpoint_overrides)

if conf.block_size < model.config.block_size:
    model.crop_block_size(conf.block_size)
    gptconf.block_size = conf.block_size

conf.model_config = gptconf
print('gptconf:', gptconf)
print('device:', device)
model.to(device)

# Optimizer setup (needed if resuming from checkpoint)
scaler = torch.cuda.amp.GradScaler(enabled=(conf.dtype == 'float16'))

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """Create AdamW optimizer with separate weight-decay groups."""
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    return optimizer

optimizer = configure_optimizers(model, conf.weight_decay, conf.learning_rate, (conf.beta1, conf.beta2), device_type)
if conf.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# DDP wrapping
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    compile = False

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# =============================================================================
# Utility functions
# =============================================================================

def load_checkpoint_with_prefix_fix(checkpoint_path, model, device='cuda'):
    """Load checkpoint, stripping DDP ('module.') and compile ('_orig_mod.') prefixes."""
    loaded_state = torch.load(checkpoint_path, map_location=device)
    state_dict = loaded_state["model"]
    unwanted_prefixes = ['_orig_mod.', 'module.']
    for prefix in unwanted_prefixes:
        for k in list(state_dict.keys()):
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    return model

def get_flops(model, inp: Union[torch.Tensor, Tuple], with_backward=False):
    """Measure FLOPs for a forward (and optionally backward) pass."""
    istrain = model.training
    model.eval()

    if isinstance(inp, tuple):
        device = next(model.parameters()).device
        vocab_size = model.config.vocab_size
        inp = torch.randint(0, vocab_size, inp, device=device)

    flop_counter = FlopCounterMode(model, display=False)
    with flop_counter:
        if with_backward:
            output = model(inp)
            logits = output[0] if isinstance(output, tuple) else output
            logits.sum().backward()
        else:
            model(inp)

    total_flops = flop_counter.get_total_flops()

    if istrain:
        model.train()
    if with_backward:
        model.zero_grad()

    return total_flops

# =============================================================================
# Wandb initialization
# =============================================================================

wandb_run_name = wandb_run_name + '_mmlu'
print('wandb_log:', conf.wandb_log, 'master_process:', master_process, 'wandb_run_name:', wandb_run_name)

unique_wandb_run_name = wandb_run_name + '_' + str(time.time())

if conf.wandb_log and master_process:
    import wandb
    config_dict_flat = {**{k: v for k, v in config_dict.items() if k != 'model_config'}, **config_dict.get('model_config', {})}
    config_dict_flat['min_lr'] = conf.get_min_lr()
    wandb.init(project=conf.wandb_project, name=wandb_run_name, config=config_dict_flat, entity=conf.wandb_entity)
    unique_wandb_run_name = wandb_run_name + '_' + wandb.run.id

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("All params:", pytorch_total_params)
    print("Trainable params:", pytorch_trainable_params)
    wandb.log({"nparams": int(pytorch_trainable_params)})

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

raw_model = model.module if ddp else model

# =============================================================================
# Inference Phase: load checkpoints, run MMLU, generate text, log to wandb
# =============================================================================

print('wandb_log:', conf.wandb_log, 'master_process:', master_process)
if conf.wandb_log and master_process:
    print("\n--- Starting Inference Phase ---")
    time.sleep(2)  # small delay for NFS propagation

    # Specify checkpoint artifact names to load
    best_ckpt_art = "Best_bs1118-cos_model=NRGPT_H_FF2W_embed=1536_depth=6_heads=12_LR=3e-05_minLR=3e-06_minLrDiv=10.0_numIter=500000_cos_pr8jetib.pt"
    last_ckpt_art = "Last_bs1118-cos_model=NRGPT_H_FF2W_embed=1536_depth=6_heads=12_LR=3e-05_minLR=3e-06_minLrDiv=10.0_numIter=500000_cos_pr8jetib.pt"

    # Log checkpoints as wandb artifacts
    best_path = os.path.join(conf.out_dir, best_ckpt_art)
    best_ckpt_art = wandb.Artifact("best_checkpoint", type="model")
    best_ckpt_art.add_file(str(best_path))
    wandb.log_artifact(best_ckpt_art)

    last_path = os.path.join(conf.out_dir, last_ckpt_art)
    last_ckpt_art = wandb.Artifact("last_checkpoint", type="model")
    last_ckpt_art.add_file(str(last_path))
    wandb.log_artifact(last_ckpt_art)

    # Instantiate fresh models and load checkpoint weights (handles DDP/compile prefixes)
    Model = load_model_from_path(conf.model_path)
    model_best = Model(conf.model_config).to(device)
    model_last = Model(conf.model_config).to(device)

    model_best = load_checkpoint_with_prefix_fix(best_path, model_best, device)
    model_best.eval()

    model_last = load_checkpoint_with_prefix_fix(last_path, model_last, device)
    model_last.eval()

    print("Models loaded successfully for generation")

    # ----- MMLU Benchmark -----
    length = 1000

    benchmark = MMLU(
        tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
        n_shots=5
    )

    enc = tiktoken.get_encoding("gpt2")
    wrapper = GPTDeepEvalWrapper(
        model=model_best,
        enc=enc,
        device='cuda',
        max_new_tokens=50,
        debug=True
    )

    benchmark.evaluate(model=wrapper)
    print(benchmark.overall_score)

    # ----- Text Generation + Quality Metrics -----
    gen_table_best = wandb.Table(columns=["run name","n_embed", "seed", "generation", "prompt", "n_params", "metrics"])
    gen_table_last = wandb.Table(columns=["run name","n_embed", "seed", "generation", "prompt", "n_params", "metrics"])

    if conf.dataset == 'openwebtext':
        enc = tiktoken.get_encoding("gpt2")
        encode = enc.encode
        decode = lambda l: enc.decode([t for t in l if t <= enc.max_token_value])

        prompts = [
            {"prompt": "\n\n","seed": 5},
            {"prompt": "In recent years, researchers have discovered that ","seed": 17},
            {"prompt": "Q: What's the best way to learn programming?\nA: ","seed": 42},
            {"prompt": "Here are 5 reasons why you should ","seed": 99},
        ]

        for prompt_seed in prompts:
            prompt, seed = prompt_seed["prompt"], prompt_seed["seed"]
            torch.manual_seed(seed)
            input_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

            torch.cuda.empty_cache()

            with torch.no_grad():
                max_length = min(length, 1024 - input_ids.shape[1])
                print('max_length:', max_length)
                generated_ids = model_best.generate(input_ids, max_new_tokens=max_length)[0]

            text = decode(generated_ids.cpu().numpy()).lstrip("\n")
            del generated_ids

            metrics = evaluate_generation(text)
            gen_table_best.add_data(wandb_run_name, conf.model_config.n_embed, seed, text, repr(prompt), pytorch_trainable_params, metrics)

            torch.cuda.empty_cache()

            with torch.no_grad():
                max_length = min(length, 1024 - input_ids.shape[1])
                generated_ids = model_last.generate(input_ids, max_new_tokens=max_length)[0]

            text = decode(generated_ids.cpu().numpy()).lstrip("\n")
            del generated_ids

            metrics = evaluate_generation(text)
            gen_table_last.add_data(wandb_run_name, conf.model_config.n_embed, seed, text, repr(prompt), pytorch_trainable_params, metrics)

            torch.cuda.empty_cache()
    else:
        # Shakespeare char-level dataset
        with open("data/shakespeare_char/input.txt", "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        print(chars)

        stoi = {ch:i for i,ch in enumerate(chars)}
        itos = {i:ch for i,ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])

        prompts = [
            {"prompt": "\n\n","seed": 5},
            {"prompt": "ROMEO: ","seed": 17},
            {"prompt": "JULIET: ","seed": 42},
            {"prompt": "To be, or not to be, that is the question:\n","seed": 99},
        ]

        for prompt_seed in prompts:
            prompt, seed = prompt_seed["prompt"], prompt_seed["seed"]
            torch.manual_seed(seed)
            input_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

            generated_ids = model_best.generate(input_ids, max_new_tokens=length)[0]
            text = decode(generated_ids.cpu().numpy()).lstrip("\n")
            metrics = evaluate_generation(text)
            gen_table_best.add_data(wandb_run_name, conf.model_config.n_embed, seed, text, repr(prompt), pytorch_trainable_params, metrics)

            generated_ids = model_last.generate(input_ids, max_new_tokens=length)[0]
            text = decode(generated_ids.cpu().numpy()).lstrip("\n")
            metrics = evaluate_generation(text)
            gen_table_last.add_data(wandb_run_name, conf.model_config.n_embed, seed, text, repr(prompt), pytorch_trainable_params, metrics)

            del generated_ids
            torch.cuda.empty_cache()

    print("Successfully saved the generations in wandb!")
    wandb.log({"gen_table_best": gen_table_best})
    wandb.log({"gen_table_last": gen_table_last})

    wandb.finish()
