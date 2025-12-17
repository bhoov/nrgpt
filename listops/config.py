from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

class Config(BaseModel):
    # wandb parameters
    project: str = Field(default=os.getenv("WANDB_PROJECT", ""))
    entity: str = Field(default=os.getenv("WANDB_ENTITY", ""))
    run_name: Optional[str] = Field(default=None, description="Name of the wandb run")
    run_name_prefix: Optional[str] = Field(default=None, description="Prefix of the wandb run, only used if run_name is not provided")
    
    model: str = Field(default="NRGPT_H_FF2W")
    model_file: Optional[str] = Field(default=None, description="Path to module defining models. should be '.py' file and importable from main script")
    n_embed: int = Field(default=8)
    n_layer: int = Field(default=1)
    n_head: int = Field(default=1)
    ff_hid_factor: int = Field(default=4, description="Hidden factor for feedforward layers, usually 4x n_embed")
    batch_size: int = Field(default=2**6)
    block_size: int = Field(default=128)
    dropout: float = Field(default=0.0)
    bias: bool = Field(default=False, description="Use bias in the model layers")
    alpha: float = 1. # step size for gradient descent
    decay_update: bool = False # If True, decay the update step size over time
    use_noise: bool = Field(default=False, description="Use noise in the model")
    use_layer_norm: bool = Field(default=True, description="Use LayerNorm in the model (only for NRGPT_AlphaGFF)")
    device: str = Field(default="cuda")
    v: int = Field(default=1, description="Version of the model")
    
    number_of_params: Optional[int] = None  # Will be filled later
    
    # Data handling
    data_file: str = Field(..., description="Path to the training and test data pickle")
    validation_split: float = Field(default=0.1)
    num_tests: int = Field(default=200, description="Number of FINAL test samples")
    num_tests_per_epoch: int = Field(default=20, description="Number of test samples at each epoch")
    save_path: str = Field(default="./")
    ops: Optional[str] = Field(default=None, description="Operations used in the data file")
    # Training parameters
    max_iters: int = Field(default=20_000)
    iter_per_batch: float = Field(default=.25, description="Number of iterations per minibatch")
    
    eval_interval: int = Field(default=100, description="Interval (# iters) run in the main training loop before evaluation")
    eval_iters: int = Field(default=10, description="Number of iterations for evaluation")
    learning_rate: float = 5e-4 #3e-4
    # device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

    min_lr: float = Field(default=1e-5, description="minimum learning rate, should be ~= learning_rate/10 per Chinchilla")
    
    ## early stopping
    early_stop: bool = Field(default=False, description="Enable early stopping")
    patience: int = Field(default=10, description="Patience for early stopping")
    min_delta: float = Field(default=2.5e-4, description="Minimum delta for early stopping")
    # min_valid_train = 5e-2
    
    vocab_size: Optional[int] = None  # Will be filled later
    gamma: float = Field(default=1.0, description="Gamma value for the model, default is 1.0")
    
    # SPECIFIC TO ATTN COMPATIBILITY for NRGPT
    tril_plus_one: bool = Field(default=False, description="Add one to the block size of attn mask of standard attention for energy compatibility")
    tril_minus_one: bool = Field(default=False, description="Subtract one from the block size of attn mask of energy attention for standard compatibility. Also resets diagonal of causal mask")

    do_generate: bool = Field(default=True, description="Generate from the model")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.vocab_size = len(self.vocab)
        # update the wandb run name
        self.run_name = self.get_run_name()
        # self.ops = self.get_ops()
        if self.tril_plus_one and self.tril_minus_one:
            raise ValueError("tril_plus_one and tril_minus_one cannot be True at the same time")
    
    def get_run_name(self):
        if self.run_name is not None: return self.run_name

        data_info = self.data_file.split('/')[-1].rstrip('_data.pkl')
        prefix = self.run_name_prefix if self.run_name_prefix is not None else ""

        return prefix + data_info + f"_model={self.model}_emb={self.n_embed}_layer={self.n_layer}_head={self.n_head}"+\
                        f"_lr={self.learning_rate}_maxiters={self.max_iters}_v={self.v}"+\
                        (f"_earlystop_patience={self.patience}_min_delta={self.min_delta}" if self.early_stop else "") 
        
    def get_ops(self):
        ops_str = self.data_file.split('/')[-1].split('funcs(')[1].split(')_depth')[0].upper()
        return '+'.join(sorted(ops_str.split(','))) # remove duplicates and sort the operations
    

def test_config():
    config = Config(data_file="data/solution_addition_vp1_data.pkl")
    print(config)
    print(f"Run name: {config.run_name}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Ops: {config.ops}")