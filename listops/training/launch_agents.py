#!/usr/bin/env python3

# see sweeps/ListOps-trip-last-100k.py for sweep_config format
# # python launch_agents.py --gpus 0 --count 600 --sweep sweeps/ListOps-trip-last-100k.py
import os
import argparse
import subprocess
import time
# from ulm_listops_sweep import sweep_config, train
import wandb
import importlib.util

import sys
sys.path.append('../')
from dotenv import load_dotenv
load_dotenv()
entity = os.getenv("WANDB_ENTITY", None)
if entity is None: raise ValueError("WANDB_ENTITY is not set")

def load_sweep_file(file_path):
    """
    Dynamically load a Python file and return its global variables
    """
    # Get the absolute path if a relative path is provided
    if not os.path.isabs(file_path):
        # Try to find the file in the current directory or sweeps directory
        if os.path.exists(file_path):
            file_path = os.path.abspath(file_path)
        elif os.path.exists(os.path.join("sweeps", file_path)):
            file_path = os.path.abspath(os.path.join("sweeps", file_path))
        else:
            raise FileNotFoundError(f"Could not find sweep file: {file_path}")
    
    # Extract the module name from the file path
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    sweep_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = sweep_module
    spec.loader.exec_module(sweep_module)
    
    # Get all variables defined in the module
    module_vars = {name: getattr(sweep_module, name) 
                    for name in dir(sweep_module) 
                    if not name.startswith('__')}
    
    return module_vars

from pprint import pprint

def main():
    parser = argparse.ArgumentParser(description='Create and launch wandb agents on multiple GPUs')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs, e.g. "0,1,2"')
    parser.add_argument('--count', type=int, default=1, help='Number of runs per GPU')
    # parser.add_argument('--project', type=str, default="ListOps-excluded-set", help='Wandb project name')
    # get sweep_id if exists
    parser.add_argument('--sweep_id', type=str, default=None, help='Wandb sweep ID to use')
    parser.add_argument('--version', type=int, default=None, help='version number for runs')
    parser.add_argument("--sweep", required=True, help="Path to the sweep configuration file")
    
    
    args = parser.parse_args()
    # Load variables from sweep file
    sweep_vars = load_sweep_file(args.sweep)
    
    # Print out key variables for demonstration
    print(f"Loaded sweep file: {args.sweep}")
    print(f"Project name: {sweep_vars.get('project_name', 'Not defined')}")
    pprint(f"Sweep config: {sweep_vars.get('sweep_config', 'Not defined')}")
    sweep_config = sweep_vars.get('sweep_config', {})
    project = sweep_vars.get('project_name', 'Not defined')
    # exit()
    
    # Create the sweep
    if args.version is not None:
        sweep_config['parameters']['v']['values'] = [args.version]

    if args.sweep_id is not None:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep=sweep_config, project=project)
    
    sweep_config['name'] = f"{project}-sweep"
    print(f"Created sweep with ID: {sweep_id}")
    
    # Launch agents on different GPUs
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(',')]
    processes = []
    
    for gpu_id in gpu_ids:
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python listops/training/agent_wandb_ulm.py --sweep_id {sweep_id} --count {args.count} "
        print(f"Launching agent on GPU {gpu_id}: {cmd}")
        
        # Using subprocess.Popen to launch in parallel
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
        
        # Small delay to prevent login conflicts
        time.sleep(3)
    
    # Wait for all processes to finish
    for p in processes:
        p.wait()

if __name__ == "__main__":
    main()