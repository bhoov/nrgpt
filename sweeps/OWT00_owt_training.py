#%%
"""
Expose `project_name` and `sweep_config` for a wandb sweep over baselines

uv run python run_sweep_wandb.py --sweep ./sweeps/OWT00_owt_training.py --num_agents 1 --allgpus
"""
import os
from dotenv import load_dotenv
load_dotenv()

entity = os.getenv("WANDB_ENTITY", None)
if entity is None: raise ValueError("WANDB_ENTITY is not set")

project_name = "NRGPT"
sweep_name = "OWT00_owt_training"

#%% Create sweep dictionary
sweep_params = {
    "batch_size": [12],
    "beta2": [0.95, 0.99],
    "bias": [False],
    "block_size": [1024],
    "compile": [True],
    "dataset": ["openwebtext"],
    "device": ["cuda"],
    "dropout": [0.0],
    "eval_interval": [1000],
    "eval_iters": [200],
    "ff_hid_factor": [4],
    "grad_clip": [1.0],
    "gradient_accumulation_steps": [4*8],
    "learning_rate": [1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 9e-5, 7e-5, 5e-5, 3e-5, 1e-5],
    "log_interval": [100],
    "lr_decay_iters": [100_000],
    "max_iters": [100_000],
    "min_lr_divfactor": [1., 1.1, 1.15, 1.2, 1.5, 2., 5., 10.],
    "model_path": ["models.NRGPT_H_FF2W", "models.GPT", "models.GPT_Rec_parallel"],
    "n_embed": [768, 1020, 1536, 2304],
    "n_head": [1, 2, 6, 12],
    "n_layer": [12],
    "out_dir": [f"out-{sweep_name}"],
    "tril_plus_one": [False],
    "wandb_entity": [entity],
    "wandb_log": [True],
    "wandb_project": [project_name],
    "wandb_run_name": [f"{sweep_name}"],
    "warmup_iters": [2000],
    "weight_decay": [0.1, 0.01],
}

## Sweep Config
sweep_config = {
    "method": "grid",  # or 'random' for random search
    "name": sweep_name, #"AlphaGFF_n_1n_8n",  # Name of the sweep
    'command': [
        'uv', 
        'run', 
        'torchrun',
        '--standalone',
        '--nproc_per_node=8',
        'train_nanogpt.py', 
        '${args}'  # This tells wandb to inject parameters as CLI args
    ],
}

sweep_config["parameters"] = {k: {"values": v} for k, v in sweep_params.items()}