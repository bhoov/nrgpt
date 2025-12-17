#%%
"""
Testing Heads against GPT pipeline

uv run python run_sweep_wandb.py --sweep ./sweeps/LOPS00_listops_training.py --num_agents 1 --gpus 0
"""
from glob import glob
import numpy as np
from dotenv import load_dotenv
load_dotenv()

project_name = "NRGPT"
sweep_name = "LOPS00_listops_training"

data_files = glob("./data/listops/List*(add_10,max*.pkl")
if not data_files: raise FileNotFoundError("No data files found")
print(f"Data files found: {data_files}")

sweep_params = {
    "data_file": data_files,
    "max_iters": [100_000],
    "early_stop": [False],
    "model": ["NRGPT_H_FF2W", "NRGPT_H_FF1", "GPT_Rec_parallel"],
    "n_embed": np.int16(np.logspace(3, 8, 16, base=2)).tolist()[::-1],
    "tril_plus_one": [True],
    "n_layer": [5],
    "ff_hid_factor": [4], # Hidden factor for feedforward layers, usually 4x n_embed
    "num_tests": [100],
    "learning_rate": [5e-4],
    "min_lr": [2e-5],
    "use_layer_norm": [True],
    "do_generate": [False],
    "run_name_prefix": [sweep_name],
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