import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList

from einops import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class RingAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()

    def forward(
        self,
        x,
        mask = None
    ):
        return x
