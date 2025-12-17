#%%
"""
uv run python run_sweep_wandb.py --sweep ./sweeps/LOPS01_listops_deep.py --num_agents 1 --gpus 0
"""
import os
from glob import glob
from dotenv import load_dotenv
load_dotenv()

entity = os.getenv("WANDB_ENTITY", None)
if entity is None: raise ValueError("WANDB_ENTITY is not set")

project_name = "NRGPT"
sweep_name = "LOPS01_listops_deep"

data_files = glob("./data/listops/List*(add_10,max*.pkl")

if not data_files: raise FileNotFoundError("No data files found")
print(f"Data files found: {data_files}")

#%% Create sweep dictionary
sweep_params = {
    "data_file": data_files,
    "max_iters": [100_000],
    "early_stop": [False], 
    "model" : [ "NRGPT_H_FF2W" ],
    "n_embed": [64],
    "n_head": [1],
    "learning_rate": [4e-4],
    "min_lr": [1e-5],
    "tril_plus_one": [True],
    "n_layer": [30],
    "ff_hid_factor": [4], # Hidden factor for feedforward layers, usually 4x n_embed
    "num_tests": [100],
    "alpha": [1.0],# could be learnable
    "use_layer_norm": [True],
    "do_generate": [True],
    "proj_type": ["pos_scalar", "identity"],
    "run_name_prefix": [sweep_name],
    "device": ["0"],
}

## Sweep Config
sweep_config = {
    "method": "grid",  # or 'random' for random search
    "name": sweep_name, #"AlphaGFF_n_1n_8n",  # Name of the sweep
    'command': [
        'uv',
        'run',
        'train_listops.py',
    ],
}

sweep_config["parameters"] = {k: {"values": v} for k, v in sweep_params.items()}