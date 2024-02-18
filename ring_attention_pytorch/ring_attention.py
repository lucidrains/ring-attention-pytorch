from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einx import rearrange

from ring_attention_pytorch.ring import (
    all_ring_pass,
    is_distributed,
    get_rank
)

from ring_attention_pytorch.ring_flash_attention import (
    ring_flash_attn
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def default_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor],
    causal: bool = False
):
    mask_value = -torch.finfo(q.dtype).max

    # similarity

    sim = einx.dot('b h i d, b h j d -> b h i j', q, k)

    # masking

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
        sim = einx.where('i j, , b h i j -> b h i j', causal_mask, mask_value, sim)

    elif exists(mask):
        sim = einx.where('b j, b h i j, -> b h i j', mask, sim, mask_value)

    # attend

    attn = einx.softmax('b h i [j]', sim)

    # aggregate

    out = einx.dot('b h i j, b h j d -> b h i d', attn, v)

    return out

# main class

class RingAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False,
        eps = 1e-10,
        q_bucket_size = 512,
        k_bucket_size = 512,
        ring_attn = False,
        ring_seq_size = 512,
        prenorm = True
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.prenorm = prenorm
        self.causal = causal

        assert divisible_by(ring_seq_size, q_bucket_size)
        assert divisible_by(ring_seq_size, k_bucket_size)

        self.ring_attn = ring_attn
        self.ring_seq_size = ring_seq_size

        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

        dim_inner = dim_head * heads
        self.to_qkv = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner * 3, bias = False)
        )

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

        device = x.device

        qkv = self.to_qkv(x)
        q, k, v = rearrange('b n (qkv h d) -> qkv b h n d', qkv, qkv = 3, h = self.heads)

        q = q * self.scale

        if not is_distributed():
            out = default_attention(q, k, v, mask = mask, causal = self.causal)
        else:
            out = ring_flash_attn(
                q, k, v,
                mask,
                causal,
                self.q_bucket_size,
                self.k_bucket_size,
                self.ring_attn
            )

        # combine heads

        out = rearrange('b h n d -> b n (h d)', out)
        return self.to_out(out)

# simple transformer for end2end testing

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

class RingTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        q_bucket_size = 512,
        k_bucket_size = 512,
        ring_attn = False,
        ring_seq_size = 512,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                RingAttention(
                    dim = dim,
                    causal = causal,
                    dim_head = dim_head,
                    heads = heads,
                    q_bucket_size = q_bucket_size,
                    k_bucket_size = k_bucket_size,
                    ring_attn = ring_attn,
                    ring_seq_size = ring_seq_size
                ),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(
        self,
        x,
        mask = None
    ):
        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.to_logits(x)
