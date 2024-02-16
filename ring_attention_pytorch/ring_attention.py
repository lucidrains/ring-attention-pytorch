import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.distributed as dist

import einx
from einx import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# ring functions

def circular_index_left(pos, ring_size):
    return ((pos - 1) + ring_size) % ring_size

def circular_index_right(pos, ring_size):
    return (pos + 1) % ring_size

# distributed ring

def circular_rank_left(rank = None, ring_size = None):
    rank = default(rank, dist.get_rank())
    ring_size = default(ring_size, dist.get_world_size())
    return circular_index_left(rank, ring_size)

def circular_rank_right(rank = None, ring_size = None):
    rank = default(rank, dist.get_rank())
    ring_size = default(ring_size, dist.get_world_size())
    return circular_index_right(rank, ring_size)

# main class

class RingAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        dim_inner = dim_head * heads
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        mask = None
    ):
        """
        einstein notation

        b - batch
        h - heads
        d - feature dimension
        n, i, j - sequence
        """

        qkv = self.to_qkv(x)
        q, k, v = rearrange('b n (qkv h d) -> qkv b h n d', qkv, qkv = 3, h = self.heads)

        q = q * self.scale

        # similarity

        sim = einx.dot('b h i d, b h j d -> b h i j', q, k)

        # masking

        mask_value = -torch.finfo(sim.dtype).max

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
            sim = einx.where('i j, , b h i j -> b h i j', causal_mask, mask_value, sim)

        elif exists(mask):
            sim = einx.where('b j, b h i j, -> b h i j', mask, sim, mask_value)

        # attend

        attn = einx.softmax('b h i [j]', sim)

        # aggregate

        out = einx.dot('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange('b h n d -> b n (h d)', out)
        return self.to_out(out)
