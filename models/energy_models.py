import math
import numbers
import torch
import torch.nn as nn
from torch.nn import functional as F
from opt_einsum import contract as einsum
from torch.func import grad
from torch.nn import LayerNorm

import sys
import os

# to allow importing from baselines.py
sys.path.append(os.path.dirname(__file__))
from baselines import GPT_Rec, FeedForward

class BareLayerNorm(nn.Module):
    """ LayerNorm but without learnable weights, only bias"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.weight = nn.Parameter(torch.ones(self.normalized_shape), requires_grad=False)

    def forward(self, x):
        return F.layer_norm(x, normalized_shape=self.normalized_shape,
                eps=self.eps, bias=self.bias, weight=self.weight)


class EnergyHead_H(nn.Module):
    """ one head of self-attention """

    def __init__(self, config, head_size=None):
        super().__init__()
        self.H = nn.Linear(config.n_embed, config.n_embed, bias=False)
        
        if config.tril_plus_one:
            # self.register_buffer('tril', torch.tril(torch.ones(config.block_size+1, config.block_size+1), diagonal=-1))
            self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size), diagonal=-1))
            # self.register_buffer('tril', torch.tril(torch.ones(config.block_size+1, config.block_size+1), diagonal=0))
        else:
            self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        xH = self.H(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = x @ xH.transpose(-2,-1) #* C**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        all_mask_rows = torch.all(self.tril[:T, :T] == 0, dim=-1, keepdim=True)  # (T, 1)
        wei = wei.masked_fill(all_mask_rows, 0.)

        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        # v = self.value(x) # (B,T,hs)
        out = wei @ xH # (B, T, T) @ (B, T, C) -> (B, T, C)
        return -out # this is grad of energy
    
    def energy(self, x):
        B,T,C = x.shape
        xH = self.H(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = x @ xH.transpose(-2,-1) #* C**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        logsumexp_wei = torch.logsumexp(wei, dim=-1)  # (B, T)
        
        all_mask_rows = torch.all(self.tril[:T, :T] == 0, dim=-1, keepdim=True)  # (T, 1)
        logsumexp_wei = logsumexp_wei.masked_fill(all_mask_rows.transpose(0,1), 0.)  # (B,T)

        # return -logsumexp_wei.sum()  # sum over all batches and time steps
        # we have a separate energy for each token and each sample in the batch
        return -logsumexp_wei  # (B,T)


class MultiHeadEnergyAttention_H(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config, head_size=None):
        super().__init__()
        self.heads = nn.ModuleList([EnergyHead_H(config) for _ in range(config.n_head)])
        # self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = 0 
        for h in self.heads:
            out = out + h(x)
        out = self.dropout(out)
        return out
    
    def energy(self, x):
        E = 0 
        for h in self.heads:
            E = E + h.energy(x)
        return E


class GradENet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = self.define_network(config)
        self.gf = grad(self.energy)
        # self.gf = torch.compile(self.gf, dynamic=True) #for ddp, doesn't work, commented out by BS
        
    def define_network(self, config):
        # Must be overriden in subclasses
        raise NotImplementedError
    
    def energy(self, x):
        return -(self.net(x)**2).sum()
    
    def forward(self, x):
        return self.gf(x)
    
        
class EnergyBlock(nn.Module):
    """ Energy Transformer block with GradFeedForward_1Lay """
    def __init__(self, config, attn_class, ffwd_class, 
                layernorm_class = nn.LayerNorm):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = config.n_embed // config.n_head
        self.attn = attn_class(config, head_size)
        self.ffwd = ffwd_class(config)
        self.ln = layernorm_class(config.n_embed)
        self.proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.scale_ff = nn.Parameter(torch.ones(1), requires_grad=True) 
        
    def forward(self, x, **kwargs):
        x = x - self.proj(self.attn(self.ln(x)) + self.scale_ff * self.ffwd(self.ln(x)))
        return x

# Simplest NRGPT with Energy Attention and 1-layer GradFF

class GradFeedForward_1Lay(GradENet):
    def __init__(self, config):
        super().__init__(config)
        
    def define_network(self, config):
        h = config.ff_hid_factor * config.n_embed # usually 4x n_embed
        return nn.Sequential(
            nn.Linear(config.n_embed, h),
            nn.GELU(),
        )

class BlockGrad_FF1Lay(EnergyBlock):
    """ Energy Transformer block with GradFeedForward_1Lay """
    def __init__(self, config):
        super().__init__(config, 
                attn_class=MultiHeadEnergyAttention_H, 
                ffwd_class=GradFeedForward_1Lay, 
                layernorm_class=BareLayerNorm)

class NRGPT_H_FF1(GPT_Rec):
    """ GPT Language Model with Energy Attention and GradFeedForward_1Lay """

    def __init__(self, config):
        super().__init__(config, block_class=BlockGrad_FF1Lay)



# NRGPT with Energy Attention and 2-layer GradFF

class GradFeedForward_2Lay(GradENet):
    def __init__(self, config):
        super().__init__(config)
        
    def define_network(self, config):
        h = config.ff_hid_factor * config.n_embed # usually 4x n_embed
        return nn.Sequential(
            nn.Linear(config.n_embed, 1*config.n_embed),# bias=config.bias),
            nn.GELU(),
            nn.Linear(1*config.n_embed,h),# bias=config.bias),
            nn.GELU(),
        )

class BlockGrad_FF2Lay(EnergyBlock):
    """ Energy Transformer block with GradFeedForward_2Lay """
    def __init__(self, config):
        super().__init__(config, 
                attn_class=MultiHeadEnergyAttention_H, 
                ffwd_class=GradFeedForward_2Lay, 
                layernorm_class=BareLayerNorm)

class NRGPT_H_FF2(GPT_Rec):
    """ GPT Language Model with Energy Attention and GradFeedForward_2Lay """

    def __init__(self, config):
        super().__init__(config, block_class=BlockGrad_FF2Lay)

        
        

# NRGPT with Energy Attention and 2-layer GradFF

class GradFeedForward_2W(GradENet):
    def __init__(self, config):
        super().__init__(config)
        
    def define_network(self, config):
        h = config.ff_hid_factor * config.n_embed # usually 4x n_embed
        return nn.Sequential(
            nn.Linear(config.n_embed, h),
            nn.GELU(),
            nn.Linear(h, config.n_embed),
        )
    
    def energy(self, x):
        return -(x*self.net(x)).sum()

class BlockGrad_FF2W(EnergyBlock):
    """ Energy Transformer block with GradFeedForward_2W """
    def __init__(self, config):
        super().__init__(config, 
                attn_class=MultiHeadEnergyAttention_H, 
                ffwd_class=GradFeedForward_2W, 
                layernorm_class=BareLayerNorm)

class NRGPT_H_FF2W(GPT_Rec):
    """ GPT Language Model with Energy Attention and GradFeedForward_2W """

    def __init__(self, config):
        super().__init__(config, block_class=BlockGrad_FF2W)
