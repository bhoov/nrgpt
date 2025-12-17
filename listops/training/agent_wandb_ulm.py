import argparse

parser = argparse.ArgumentParser(description='Launch wandb agents for ListOps sweep')
parser.add_argument('--count', type=int, default=1, help='Number of runs per GPU')
parser.add_argument('--sweep_id', type=str, default=None, help='Wandb sweep ID to use')

args = parser.parse_args()

import wandb

# from ulm_listops_sweep import train
from listops.training.train import train
wandb.agent(args.sweep_id, train, count=args.count)