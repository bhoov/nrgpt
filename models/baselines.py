#%%
"""
Full definition of our Energy GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from opt_einsum import contract as einsum
from torch.func import grad
from torch.nn import LayerNorm


### Base components for the transformer
#--------------------------------------------------------------------

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embed, head_size, bias=False)
        self.query = nn.Linear(config.n_embed, head_size, bias=False)
        self.value = nn.Linear(config.n_embed, head_size, bias=False)

        if config.tril_plus_one:
            # self.register_buffer('tril', torch.tril(torch.ones(config.block_size+1, config.block_size+1), diagonal=-1))
            self.register_buffer('tril', torch.tril(torch.ones(config.block_size+1, config.block_size+1), diagonal=0))
        else:
            self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        all_mask_rows = torch.all(self.tril[:T, :T] == 0, dim=-1, keepdim=True)  # (T, 1)
        wei = wei.masked_fill(all_mask_rows, 0.)

        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(head_size * config.n_head, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        h = config.ff_hid_factor * config.n_embed # usually 4x n_embed
        self.net = nn.Sequential(
            nn.Linear(config.n_embed,h),
            nn.ReLU(),
            nn.Linear(h, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


#%% Standard blocks
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = config.n_embed // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x, **kwargs):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Block_parallel(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = config.n_embed // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)


    def forward(self, x, **kwargs):
        x = x + self.sa(self.ln1(x)) + self.ffwd(self.ln2(x))
        return x



class GPT(nn.Module):
    def __init__(self, config, block_class=Block):
        super().__init__()
        self.config = config
        self.block_class = block_class
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.register_buffer('pos_idx', torch.arange(config.block_size)) # (block_size,)
        # self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.blocks = self.get_blocks(config)
        self.ln_f = nn.LayerNorm(config.n_embed) # final layer norm
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_blocks(self, config):
        return nn.Sequential(*[self.block_class(config) for _ in range(config.n_layer)])
    
    def block_forward(self, x):
        return self.blocks(x)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # torch.arange(T, device=device)
        pos_emb = self.position_embedding_table(self.pos_idx[:T]) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        # x = self.blocks(x) # (B,T,C)
        x = self.block_forward(x)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, greedy=False):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            if not greedy:
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)  # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                # print(idx_next.shape, logits.shape)
            else:
                # greedy shortcut -- just use argmax of probs/logits
                idx_next = torch.argmax(logits, dim=1).unsqueeze(-1)
                # print(idx_next.shape, logits.shape)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class GPT_Rec(GPT):
    """ Recursive transformer where the same block is applied n_layer times """
    def __init__(self, config, block_class=Block):
        super().__init__(config, block_class=block_class)
        
    def get_blocks(self, config):
        # Only a single block that will be applied recursively
        return self.block_class(config)
    
    def block_forward(self, x):
        B, T, C = x.shape
        for _ in range(self.config.n_layer):
            x = self.blocks(x)
        return x

class GPT_Rec_parallel(GPT_Rec):
    """ Variant of recursive GPT using Block_parallel """
    def __init__(self, config, block_class=Block_parallel):
        super().__init__(config, block_class=block_class)
        