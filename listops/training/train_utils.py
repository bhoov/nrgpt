import os
import importlib.util
import sys
import wandb
from typing import Callable

def get_model_class(config, Model_default) -> Callable:
    if config.model_file is not None:
        print(f"Loading model class {config.model} from {config.model_file}")
        
        if os.path.isfile(config.model_file):
            # Load from a specific file
            spec = importlib.util.spec_from_file_location("Model", config.model_file)
            Models = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(Models)
            return getattr(Models, config.model)
            
        elif os.path.isdir(config.model_file):
            # Load from a package directory
            # Normalize the path to handle trailing slashes
            normalized_path = os.path.normpath(os.path.abspath(config.model_file))
            parent_dir = os.path.dirname(normalized_path)
            package_name = os.path.basename(normalized_path)
            
            if not package_name:
                raise ValueError(f"Could not determine package name from path: {config.model_file}")
            
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
                added_to_path = True
            else:
                added_to_path = False
            
            try:
                # Import the package
                Models = importlib.import_module(package_name)
                # Force reload to ensure we get the latest version
                importlib.reload(Models)
                return getattr(Models, config.model)
            finally:
                # Clean up sys.path if we added to it
                if added_to_path:
                    sys.path.remove(parent_dir)
        else:
            raise FileNotFoundError(f"Model path {config.model_file} does not exist or is neither a file nor directory.")
    else:
        print(f"Loading model class {config.model} from built-in models.")
        return getattr(Model_default, config.model)



import json
def save_data_artifact(model_config, data):
    
        # Get filename for artifact naming
        data_filename = os.path.basename(model_config.data_file)
        # remove special characters from the filename
        data_filename = ''.join(c for c in data_filename if c.isalnum() or c in ('_', '-')).rstrip('.pkl')
        
        # Check if this dataset already exists as an artifact
        try:
            # Try to use the existing artifact if available
            artifact = wandb.use_artifact(f"listops-dataset-{data_filename}:latest", type="dataset")
            print(f"Using existing dataset artifact: {artifact.name}")
        except Exception:
            # If not found, create and log a new artifact
            print(f"Creating new dataset artifact for {data_filename}")
            dataset_artifact = wandb.Artifact(
                name=f"listops-dataset-{data_filename}",
                type="dataset",
                description="ListOps dataset with metadata"
            )
            
            # Add the data file to the artifact
            dataset_artifact.add_file(model_config.data_file)
            
            # Add metadata as JSON
            metadata = data['metadata']
            
            metadata_path = os.path.join(model_config.save_path, f"dataset_metadata_{data_filename}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            dataset_artifact.add_file(metadata_path)
            
            # Log the artifact to wandb - this uploads it once
            artifact = wandb.log_artifact(dataset_artifact)
        
        # Link the artifact to this run - this doesn't upload again, just creates a reference
        wandb.run.use_artifact(artifact)
        
        # Add the artifact name to the config for reference
        wandb.config.update({"dataset_artifact": artifact.name})
        
        
def strip_num(func_name):
    """Strip the number from the function name."""
    if '_' in func_name: 
        return '_'.join(func_name.split('_')[:-1])
    else:
        return func_name