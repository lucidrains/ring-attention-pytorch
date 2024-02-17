import torch
from torch import nn
from torch.nn import Module, ModuleList

import einx
from einx import rearrange

from ring_attention_pytorch.ring import (
    all_ring_pass,
    is_distributed,
    get_rank
)

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
        heads = 8,
        causal = False,
        eps = 1e-10
    ):
        super().__init__()
        self.eps = eps
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
        device, mask_value = x.device, -torch.finfo(x.dtype).max

        qkv = self.to_qkv(x)
        q, k, v = rearrange('b n (qkv h d) -> qkv b h n d', qkv, qkv = 3, h = self.heads)

        q = q * self.scale

        if not is_distributed():
            # similarity

            sim = einx.dot('b h i d, b h j d -> b h i j', q, k)

            # masking

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

        else:
            # accumulate outputs numerator and denominator

            out = torch.zeros_like(q)
            row_sums = torch.zeros((*q.shape[:-1], 1), device = device)
            row_maxes = torch.full((*q.shape[:-1], 1), mask_value, device = device)

            row_chunk_size = q.shape[-2]
            col_chunk_size = k.shape[-2]
            row_offset = get_rank() * row_chunk_size

            for ring_rank, (k, v, mask) in all_ring_pass(k, v, mask):

                col_offset = col_chunk_size * ring_rank

                attn_weights = einx.dot('... i d, ... j d -> ... i j', q, k)

                if self.causal:
                    i, j = attn_weights.shape[-2:]
                    causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + row_offset - col_offset + 1)
                    attn_weights = einx.where('i j, b h i j, -> b h i j', causal_mask, attn_weights, mask_value)

                elif exists(mask):
                    attn_weights = einx.where('b j, b h i j, -> b h i j', mask, attn_weights, mask_value)

                block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_weights = torch.exp(attn_weights - new_row_maxes)

                if self.causal:
                    exp_weights = einx.where('i j, b h i j,', causal_mask, exp_weights, 0.)
                elif exists(mask):
                    exp_weights = einx.where('b j, b h i j,', mask, exp_weights, 0.)

                block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = self.eps)

                exp_values = einx.dot('... i j, ... j d -> ... i d', exp_weights, v)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)

                # update row sums and row maxes

                row_sums = exp_row_max_diff * row_sums + block_row_sums
                row_maxes = new_row_maxes

                # accumulate out

                out = out * exp_row_max_diff + exp_values

            out = out / row_sums

        # combine heads

        out = rearrange('b h n d -> b n (h d)', out)
        return self.to_out(out)
