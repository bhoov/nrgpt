import os
import argparse
import subprocess
import time
import wandb
import importlib.util
import sys
from pprint import pprint

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


def load_sweep_from_id(sweep_id, project_name=None, entity=None):
    """
    Load sweep configuration from a wandb sweep ID
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # If project name is not provided, try to extract from sweep_id format
    # sweep_id format is usually: entity/project/sweep_id or just sweep_id
    if project_name is None:
        # Try to get the sweep directly if it's a full path
        try:
            sweep = api.sweep(f"{entity}/{project_name}/{sweep_id}")
        except Exception as e:
            raise ValueError(f"Could not load sweep {sweep_id}. You may need to provide --project argument. Error: {e}")
    else:
        # Construct the full sweep path
        sweep_path = f"{project_name}/{sweep_id}" if '/' not in sweep_id else sweep_id
        sweep = api.sweep(f"{entity}/{sweep_path}")
    
    # Extract the configuration
    sweep_config = sweep.config
    project_name = sweep.project
    
    # Return in the same format as load_sweep_file
    return {
        'project_name': project_name,
        'sweep_config': sweep_config,
        'sweep_name': sweep.name or sweep_id
    }


def main():
    parser = argparse.ArgumentParser(description='Create and launch wandb agents on multiple GPUs')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs, e.g. "0,1,2"')
    parser.add_argument('--allgpus', action='store_true', help='Use all available GPUs')
    # parser.add_argument('--count', type=int, default=1, help='Number of runs per GPU')
    parser.add_argument('--num_agents', type=int, default=1, help='Number of wandb agents to launch per GPU')
    # parser.add_argument('--project', type=str, default="ListOps-excluded-set", help='Wandb project name')
    # get sweep_id if exists
    parser.add_argument('--sweep_id', type=str, default=None, help='Wandb sweep ID to use')
    parser.add_argument('--project', type=str, default=None, help='Wandb project name (required when using sweep_id)')
    parser.add_argument('--entity', type=str, default=None, help='Wandb entity name (required when using sweep_id)')
    parser.add_argument('--version', type=int, default=None, help='version number for runs')
    parser.add_argument("--sweep", default=None, help="Path to the sweep configuration file")
    args = parser.parse_args()

    if args.sweep is None and args.sweep_id is None:
        raise ValueError("Either --sweep (file path) or --sweep_id is required")

    # Load variables from sweep file OR sweep ID
    if args.sweep_id is not None:
        print(f"Loading sweep from ID: {args.sweep_id}")
        sweep_vars = load_sweep_from_id(args.sweep_id, args.project, args.entity)
        print(f"Loaded sweep: {sweep_vars.get('sweep_name', args.sweep_id)}")
    else:
        print(f"Loading sweep from file: {args.sweep}")
        sweep_vars = load_sweep_file(args.sweep)
        print(f"Loaded sweep file: {args.sweep}")

    project = sweep_vars.get('project_name', 'Not defined')
    sweep_config = sweep_vars.get('sweep_config', {})

    print(f"Project name: {project}")
    pprint(f"Sweep config: {sweep_config}")

    # Create the sweep or use existing one
    if args.version is not None:
        sweep_config['parameters']['v']['values'] = [args.version]

    if args.sweep_id is not None:
        sweep_id = args.sweep_id
        print(f"Using existing sweep with ID: {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep=sweep_config, project=project)
        print(f"Created new sweep with ID: {sweep_id}")
    
    sweep_config['name'] = f"{project}-sweep"

    def make_cmd():
        cmd = f"wandb agent --project {project}"
        if args.entity is not None: cmd += f" --entity {args.entity}"
        cmd += f" {sweep_id}"
        return cmd

    processes = []
    if args.allgpus:
        for i in range(args.num_agents):
            cmd = make_cmd()
            print(f"Launching agent {i+1}/{args.num_agents} on ALL GPUs: {cmd}")
            process = subprocess.Popen(cmd, shell=True)
            processes.append(process)
            time.sleep(2)
    else:
        gpu_ids = [gpu.strip() for gpu in args.gpus.split(',')]
        for gpu_id in gpu_ids:
            for i in range(args.num_agents):
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = gpu_id
                cmd = f"wandb agent --project {project}"
                if args.entity is not None: cmd += f" --entity {args.entity}"
                cmd += f" {sweep_id}"

                print(f"Launching agent {i+1}/{args.num_agents} on GPU {gpu_id}: {cmd}")
            
                # Using subprocess.Popen to launch in parallel
                process = subprocess.Popen(cmd, shell=True, env=env)
                processes.append(process)
                
                # Small delay to prevent login conflicts
                time.sleep(3)
            
    # Wait for all processes to finish
    for p in processes:
        p.wait()

if __name__ == "__main__":
    main()