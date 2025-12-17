"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

New flags:
    uv run python train_nanogpt.py --model_path=models.NRGPT_H_FF2W
    uv run python train_nanogpt.py --model_path=models.GPT
"""
#%%
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import tyro
import copy
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

print('check', torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

@tyro.conf.configure(tyro.conf.FlagConversionOff, tyro.conf.OmitArgPrefixes)
@attrs.define
class TrainingConfig:
    # I/O
    out_dir: str = 'out'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False # if True, script exits right after the first eval
    init_from: str = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    
    # wandb logging
    wandb_log: bool = False # disabled by default
    wandb_project: str = 'owt'
    wandb_run_name: str = 'gpt2' #'run' + str(time.time())
    wandb_entity: Optional[str] = None

    # optimizer
    learning_rate: float = 6e-4 # max learning rate
    max_iters: int = 600000 # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
    
    # learning rate decay
    decay_lr: bool = True # whether to decay the learning rate
    cosine_decay: bool = False # If false use exp decay
    warmup_iters: int = 2000 # how many steps to warm up for
    lr_decay_iters: int = 600000 # should be ~= max_iters per Chinchilla
    min_lr: Optional[float] = None # minimum learning rate. If None, will be learning_rate/10 per Chinchilla
    min_lr_divfactor: float = 10. # how much to divide the learning rate by to get the min learning rate. Overridden if min_lr is specified
    
    # data
    dataset: str = 'openwebtext'
    gradient_accumulation_steps: int = 5*8 # used to simulate larger batch sizes
    batch_size: int = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size

    # DDP settings
    backend: str = 'nccl' # 'nccl', 'gloo', etc.
    
    # system
    device: str = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile: bool = True # use PyTorch 2.0 to compile the model to be faster

    # Configuration options for the actual model
    model_config: ModelConfig = ModelConfig()
    
    # model type
    model_path: str = "models.NRGPT_H_FF2W"  # e.g., "models.NRGPT_H_FF2W", "models.GPT"

    def get_wandb_run_name(self) -> str:
        """Generate wandb run name based on model type and hyperparameters"""
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
    """Load config from .py, .json, or .yaml file and merge with base config"""
    path = Path(config_file)
    
    if path.suffix == '.py':
        config_globals = {}
        exec(path.read_text(), config_globals)
        # Filter to only config-relevant keys (exclude builtins, imports, etc.)
        overrides = {k: v for k, v in config_globals.items() if not k.startswith('_')}
        
    elif path.suffix in ['.json']: overrides = json.loads(path.read_text())
    elif path.suffix in ['.yaml', '.yml']: overrides = yaml.safe_load(path.read_text())
    else: raise ValueError(f"Unsupported config file format: {path.suffix}")

    # Parse out the ModelConfig args, if present
    model_config_fields = set(attrs.fields_dict(ModelConfig).keys())
    training_config_fields = set(attrs.fields_dict(TrainingConfig).keys())
    
    # Separate model config fields from training config fields
    model_config_overrides = {k: v for k, v in overrides.items() if k in model_config_fields}
    training_config_overrides = {k: v for k, v in overrides.items() if k in training_config_fields}
    
    # Create nested structure if we have model config overrides
    if model_config_overrides:
        # Start with default model config and update with overrides
        base_model_config = unstructure(ModelConfig())
        base_model_config.update(model_config_overrides)
        training_config_overrides['model_config'] = base_model_config

    return structure(training_config_overrides, TrainingConfig)
    
# Parse configuration using tyro
if is_interactive(): 
    conf = TrainingConfig()
else: 
    # Check if first argument is a config file
    conf = TrainingConfig()

    config_file = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        config_file = sys.argv[1]
        # Remove config file from sys.argv so tyro doesn't see it
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        # Apply config file overrides first
        print("config_file:", config_file)
        conf = load_config_from_file(config_file)
    
    # Parse with tyro (gets CLI overrides)
    conf = tyro.cli(TrainingConfig, default=conf)

conf: TrainingConfig = conf # For type hinting
config_dict = unstructure(conf)
print("config_dict:", config_dict)
print("conf:", conf)

#%%
# set wandb_run_name including the basic hyperparameters
# print('wandb_run_name before:', wandb_run_name)
wandb_run_name = conf.get_wandb_run_name()
print('wandb_run_name:', wandb_run_name)
print("dataset:", conf.dataset)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
print('rank:', int(os.environ.get('RANK', -1)), 'ddp:', ddp)
if ddp:
    init_process_group(backend=conf.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert conf.gradient_accumulation_steps % ddp_world_size == 0
    conf.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    device = conf.device
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = conf.gradient_accumulation_steps * ddp_world_size * conf.batch_size * conf.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(conf.out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 onwarmup_iters cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[conf.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', conf.dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
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
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
# meta = pickle.load(open(meta_path, 'rb'))
# meta_vocab_size = meta['vocab_size']
# print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")



def load_model_from_path(model_path: str):
    """Load model class from `path.to.module.ClassName` path"""
    if '.' not in model_path:
        raise ValueError(f"model_path must be in 'path.to.module.ClassName' format, got: {model_path}")
    
    module_name, class_name = model_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

gptconf = conf.model_config
Model = load_model_from_path(conf.model_path)

if conf.init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None: print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    gptconf.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304 #50304
    model = Model(gptconf)
    model_init = Model(gptconf)
elif conf.init_from == 'resume':
    print(f"Resuming training from {conf.out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(conf.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    gptconf = attrs.evolve(gptconf, **checkpoint_model_args)
    model = Model(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif conf.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {conf.init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=conf.model_config.dropout)
    model = Model.from_pretrained(conf.init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    checkpoint_overrides = {}
    for k in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias', 'vocab_size']:
        checkpoint_overrides[k] = getattr(model.config, k)
    gptconf = attrs.evolve(gptconf, **checkpoint_overrides)
# crop down the model block size if desired, using model surgery
if conf.block_size < model.config.block_size:
    model.crop_block_size(conf.block_size)
    gptconf.block_size = conf.block_size # so that the checkpoint will have the right value

conf.model_config = gptconf # Update
print('gptconf:', gptconf)
print('device:', device)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(conf.dtype == 'float16'))

# optimizer
import inspect
def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
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
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

optimizer = configure_optimizers(model, conf.weight_decay, conf.learning_rate, (conf.beta1, conf.beta2), device_type)
if conf.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# wrap model into DDP container
if ddp:
    # model = DDP(model, device_ids=[ddp_local_rank]) #original one
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True) # by BS
    compile = False # by BS
    
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(conf.eval_iters)
        for k in range(conf.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr_cosine(it):
    # 1) linear warmup for warmup_iters steps
    if it < conf.warmup_iters:
        return conf.learning_rate * (it + 1) / (conf.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > conf.lr_decay_iters:
        return conf.get_min_lr()
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - conf.warmup_iters) / (conf.lr_decay_iters - conf.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return conf.get_min_lr() + coeff * (conf.learning_rate - conf.get_min_lr())

# learning rate decay scheduler (exponential with warmup)
def get_lr_exp(it):
    # 1) linear warmup for warmup_iters steps
    if it < conf.warmup_iters:
        return conf.learning_rate * (it + 1) / (conf.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > conf.lr_decay_iters:
        return conf.get_min_lr()
    # 3) in between, use exponential decay down to min learning rate
    decay_ratio = (it - conf.warmup_iters) / (conf.lr_decay_iters - conf.warmup_iters)
    assert 0 <= decay_ratio <= 1
    
    # Exponential interpolation between learning_rate and min_lr
    # When decay_ratio=0: returns learning_rate, when decay_ratio=1: returns min_lr
    log_lr = math.log(conf.learning_rate) * (1 - decay_ratio) + math.log(conf.get_min_lr()) * decay_ratio
    return math.exp(log_lr)

# logging
print('wandb_log:', conf.wandb_log, 'master_process:', master_process)

# Use the current time to make the run name unique if not using wandb
unique_wandb_run_name = wandb_run_name + '_' + str(time.time())

if conf.wandb_log and master_process:
    import wandb
    config_dict_flat = {**{k: v for k, v in config_dict.items() if k != 'model_config'}, **config_dict.get('model_config', {})}
    config_dict_flat['min_lr'] = conf.get_min_lr() # Update with manual min_lr
    wandb.init(project=conf.wandb_project, name=wandb_run_name, config=config_dict_flat, entity=conf.wandb_entity)
    unique_wandb_run_name = wandb_run_name + '_' + wandb.run.id

    # total params (all)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    # trainable params only
    pytorch_trainable_params = sum(
        p.numel() for p in model.parameters() 
        if p.requires_grad
    )
    print("All params:", pytorch_total_params)
    print("Trainable params:", pytorch_trainable_params)
    wandb.log({"nparams": int(pytorch_trainable_params)})

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    if conf.decay_lr:
        lr = get_lr_cosine(iter_num) if conf.cosine_decay else get_lr_exp(iter_num)
    else:
        lr = conf.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % conf.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print("losses:", losses)
        print(f"step {iter_num/conf.eval_interval}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if conf.wandb_log:

            log_data = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }

            # Check if alpha is a learnable parameter
            model_to_check = model.module if ddp else model
            if hasattr(model_to_check, '_orig_mod'): model_to_check = model_to_check._orig_mod
            model_state = model.module.state_dict() if ddp else model.state_dict()

            # Look for alpha in named parameters
            for name, param in model_to_check.named_parameters():
                if 'alpha' in name or "scale_ff" in name:
                    log_data["alpha"] = param.item()
                    break  # Take the first alpha or scale_ff found
            
            wandb.log(log_data)
            
        if losses['val'] < best_val_loss: 
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': unstructure(conf.model_config),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config_dict,
                }

                print(f"New best validation model at step {iter_num/conf.eval_interval} with loss {best_val_loss:0.04f}")
                torch.save(checkpoint, os.path.join(conf.out_dir, 'Best_' + unique_wandb_run_name + '.pt'))


    if iter_num == 0 and conf.eval_only:
        break


    # # print which one does not contribute loss
    # for i, (name, param) in enumerate(model.named_parameters()):
    #     if i == 5:
    #         print(f"Parameter {i}: {name}")
    
    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         print(f"{name} did not receive grad")

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(conf.gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == conf.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / conf.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradientx
    if conf.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % conf.log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * conf.gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            if hasattr(raw_model, 'estimate_mfu'):
                mfu = raw_model.estimate_mfu(conf.batch_size * conf.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            else:
                running_mfu = -1.0
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > conf.max_iters:
        break

# saving the last training model
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': unstructure(conf.model_config),
    'iter_num': iter_num,
    'config': config_dict,
}
print("Saving the last iteration model")
torch.save(checkpoint, os.path.join(conf.out_dir, 'Last_' + unique_wandb_run_name + '.pt'))


if ddp and master_process:
    print('Training complete. Shutting down the DDP process group.')
    destroy_process_group()



#------------------------------Inference Phase------------------------------
print('wandb_log:', conf.wandb_log, 'master_process:', master_process)
if conf.wandb_log and master_process:
    print("\n--- Starting Inference Phase ---")
    # Log best checkpoint to wandb and generate output
    best_path = os.path.join(conf.out_dir, 'Best_' + unique_wandb_run_name + '.pt')
    best_ckpt_art = wandb.Artifact("best_checkpoint", type="model")
    best_ckpt_art.add_file(str(best_path))
    wandb.log_artifact(best_ckpt_art)

    # Log last checkpoint to wandb and generate output
    last_path = os.path.join(conf.out_dir, 'Last_' + unique_wandb_run_name + '.pt')
    last_ckpt_art = wandb.Artifact("last_checkpoint", type="model")
    last_ckpt_art.add_file(str(last_path))
    wandb.log_artifact(last_ckpt_art)

    #Load back whichever you need
    model_best = copy.deepcopy(raw_model)
    model_last = copy.deepcopy(raw_model)
    # model_best = copy.deepcopy(model_init)
    # model_last = copy.deepcopy(model_init)

    # load the best model
    loaded_state = torch.load(best_path, map_location=device) 
    model_best.load_state_dict(loaded_state["model"])
    model_best.eval()

    # load the last model
    loaded_state = torch.load(last_path, map_location=device) 
    model_last.load_state_dict(loaded_state["model"])
    model_last.eval()

    print("Models loaded successfully for generation")

    # Generate 4 random samples of 1000 tokens
    n_samples = 4
    length = 1000

    # prepare a W&B table
    gen_table_best = wandb.Table(columns=["run name","n_embed", "seed", "generation", "prompt", "n_params", "metrics"])
    gen_table_last = wandb.Table(columns=["run name","n_embed", "seed", "generation", "prompt", "n_params", "metrics"])

    if conf.dataset == 'openwebtext':
        # Load the pre-trained GPT-2 tokenizer from tiktoken with the BPE logic.
        enc = tiktoken.get_encoding("gpt2")
        encode = enc.encode
        # decode = lambda l: enc.decode(l) # works for only vocab size <= 50257
        decode = lambda l: enc.decode([t for t in l if t <= enc.max_token_value]) # works for any vocab size

        prompts = [
            {"prompt": "\n\n","seed": 5},
            {"prompt": "In recent years, researchers have discovered that ","seed": 17},
            {"prompt": "Q: What's the best way to learn programming?\nA: ","seed": 42},
            {"prompt": "Here are 5 reasons why you should ","seed": 99},
        ]

        # Just add these lines to your existing code
        for prompt_seed in prompts:
            prompt, seed = prompt_seed["prompt"], prompt_seed["seed"]
            torch.manual_seed(seed)
            input_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            
            # Clear cache before each generation
            torch.cuda.empty_cache()
            
            # Use no_grad context
            with torch.no_grad():
                # Limit generation length to 1000
                max_length = min(length, 1024 - input_ids.shape[1])
                print('max_length:', max_length)
                generated_ids = model_best.generate(input_ids, max_new_tokens=max_length)[0]
            
            text = decode(generated_ids.cpu().numpy()).lstrip("\n")
            # Delete tensor immediately
            del generated_ids
            
            metrics = evaluate_generation(text)
            gen_table_best.add_data(wandb_run_name, conf.model_config.n_embed, seed, text, repr(prompt), pytorch_trainable_params, metrics)
            
            # Clear cache between models
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                max_length = min(length, 1024 - input_ids.shape[1])
                generated_ids = model_last.generate(input_ids, max_new_tokens=max_length)[0]
            
            text = decode(generated_ids.cpu().numpy()).lstrip("\n")
            del generated_ids
            
            metrics = evaluate_generation(text)
            gen_table_last.add_data(wandb_run_name, conf.model_config.n_embed, seed, text, repr(prompt), pytorch_trainable_params, metrics)
            
            # Clear after each prompt
            torch.cuda.empty_cache()
    else:
        with open("data/shakespeare_char/input.txt", "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        print(chars)

        # create a mapping from characters to integers
        stoi = {ch:i for i,ch in enumerate(chars)}
        itos = {i:ch for i,ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        decode = lambda l: "".join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        prompts = [
            {"prompt": "\n\n","seed": 5},
            {"prompt": "ROMEO: ","seed": 17},
            {"prompt": "JULIET: ","seed": 42},
            {"prompt": "To be, or not to be, that is the question:\n","seed": 99},
        ]
    
        for prompt_seed in prompts:
            prompt, seed = prompt_seed["prompt"], prompt_seed["seed"]
            torch.manual_seed(seed)
            # tokenize & move to device; shape (1, prompt_len)
            input_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            
            generated_ids = model_best.generate(input_ids, max_new_tokens=length)[0]
            text = decode(generated_ids.cpu().numpy()).lstrip("\n")
            metrics = evaluate_generation(text)
            # print('model_best: prompt:',  prompt, 'metrics:', metrics)
            gen_table_best.add_data(wandb_run_name, conf.model_config.n_embed, seed, text, repr(prompt), pytorch_trainable_params, metrics)

            # generate with the last model
            generated_ids = model_last.generate(input_ids, max_new_tokens=length)[0]
            text = decode(generated_ids.cpu().numpy()).lstrip("\n")
            metrics = evaluate_generation(text)
            # print('model_last: prompt:',  prompt, 'metrics:', metrics)
            gen_table_last.add_data(wandb_run_name, conf.model_config.n_embed, seed, text, repr(prompt), pytorch_trainable_params, metrics)



    print("Successfully saved the generations in wandb!")
    wandb.log({"gen_table_best": gen_table_best})
    wandb.log({"gen_table_last": gen_table_last})

    wandb.finish()