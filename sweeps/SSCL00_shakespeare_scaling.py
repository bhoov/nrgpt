#%%
"""
Expose `project_name` and `sweep_config` for a wandb sweep over baselines

uv run python run_sweep_wandb.py --sweep ./sweeps/SSCL00_shakespeare_scaling.py --num_agents 1 --gpus 0
"""
import os
from dotenv import load_dotenv
load_dotenv()

entity = os.getenv("WANDB_ENTITY", None)
if entity is None: raise ValueError("WANDB_ENTITY is not set")

project_name = "NRGPT"
sweep_name = "SSCL00_shakespeare_scaling"

#%% Create sweep dictionary
sweep_params = {
    "batch_size": [64],
    "beta2": [0.95, 0.99],
    "bias": [False],
    "block_size": [256],
    "compile": [True],
    "dataset": ["shakespeare_char"],
    "device": ["cuda"],
    "dropout": [0.1],
    "eval_interval": [250],
    "eval_iters": [200],
    "ff_hid_factor": [4],
    "grad_clip": [1.0],
    "gradient_accumulation_steps": [1],
    "learning_rate": [1e-3, 5e-5, 1e-4],
    "log_interval": [100],
    "lr_decay_iters": [40_000],
    "max_iters": [40_000],
    "min_lr_divfactor": [10., 20., 100.],
    "model_path": ["models.NRGPT_H_FF2W", "models.NRGPT_H_FF2", "models.GPT", "models.GPT_Rec_parallel"],
    "n_embed": [32, 64, 128, 256, 380, 512, 768],
    "n_head": [1, 2, 4, 8],
    "n_layer": [3, 6, 8],
    "out_dir": [f"out-{sweep_name}"],
    "tril_plus_one": [False],
    "wandb_entity": [entity],
    "wandb_log": [True],
    "wandb_project": [project_name],
    "wandb_run_name": [f"{sweep_name}"],
    "warmup_iters": [100],
    "weight_decay": [0.1, 0.01],
}

## Sweep Config
sweep_config = {
    "method": "grid",  # or 'random' for random search
    "name": sweep_name, #"AlphaGFF_n_1n_8n",  # Name of the sweep
    'command': [
        'uv', 
        'run', 
        'train_nanogpt.py',  # Path to your training script
        '${args}'  # This tells wandb to inject parameters as CLI args
    ],
}

sweep_config["parameters"] = {k: {"values": v} for k, v in sweep_params.items()}