from attrs import define
from typing import Optional

# This is config is shared across *all* models
@define
class ModelConfig():
    block_size: int = 1024 # Maximum sequence length
    n_embed: int = 768
    n_layer: int = 12
    n_head: int = 12
    ff_hid_factor: int = 4 # Hidden factor for feedforward layers, usually 4x n_embed
    dropout: float = 0.0
    bias: bool = False # Use bias in the model layers
    vocab_size: Optional[int] = None  # Will be filled later

    tril_plus_one: bool = False # Add one to the block size of attn mask of standard attention for energy compatibility. Used in energy models
    alpha: float = 1.0 # Step size for gradient descent. Used for energy
    decay_update: bool = False # If True, decay the update step size over time. Used for energy
    use_layer_norm: bool = True # If False, use identity instead of standard layer norm. Used for energy
