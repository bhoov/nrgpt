# train owt model
"""
uv run python train_nanogpt.py config/train_owt_config.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

wandb_log = True
wandb_entity = os.getenv("WANDB_ENTITY", None)
if wandb_entity is None: raise ValueError("WANDB_ENTITY is not set")

out_dir = 'out-owt-nrgpt'
eval_interval = 1000 #keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 #don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

dataset = 'openwebtext'
gradient_accumulation_steps = 4*8

batch_size = 12 
block_size = 1024

n_layer = 12 
n_head = 12
n_embed = 768
dropout = 0.0

learning_rate = 1e-4
max_iters = 100000 
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 1e-4/2
weight_decay = 1e-1
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 2000

# device = 'cuda:7' # run on cpu only

# set learning rate scheduler either to cosine or exp decay
cosine_decay = True
if cosine_decay: suffix='cos'
else: suffix='exp'

model_path = "models.NRGPT_H_FF2W" 
wandb_run_name=suffix

