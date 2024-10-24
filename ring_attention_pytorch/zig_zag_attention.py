from math import ceil

import torch
from torch import nn, tensor
import torch.nn.functional as F

from einops import einsum, rearrange, repeat, reduce

from ring_attention_pytorch.distributed import (
    all_gather_variable_dim,
    AllGather
)

from ring_attention_pytorch.ring import (
    get_rank,
    get_world_size
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# pad sequence to 2 x <world size> for sharding

def zig_zag_pad_seq(t):
    seq_len = t.shape[-2]
    chunks = 2 * get_world_size()

    padded_seq_len = ceil(seq_len / chunks) * chunks
    t = F.pad(t, (0, 0, 0, padded_seq_len - seq_len), value = 0.)

    def inverse(out):
        return out[..., :seq_len, :]

    return t, inverse

# zig zag sharding and its inverse

def zig_zag_shard(t, all_gather_batch = False):
    rank, world_size = get_rank(), get_world_size()

    if all_gather_batch:
        all_gather = AllGather(dim = 0)
        t, gather_sizes = all_gather(t)

    chunks = 2 * world_size
    t = t.chunk(chunks, dim = -2)

    # each rank takes care of two chunks for their simple workload balancing scheme

    two_chunks = torch.cat((t[rank], t[chunks - 1 - rank]), dim = -2)

    # inverse

    def inverse(two_chunks):

        two_chunks = rearrange(two_chunks, '... (two n) d -> ... two n d', two = 2)
        all_gather = AllGather(dim = -2)
        all_chunks, _ = all_gather(two_chunks)

        first_half, second_half = rearrange(all_chunks, '... two n d -> two ... n d')

        out = torch.cat((first_half, second_half.flip(dims = (-2,))), dim = -2)

        if all_gather_batch:
            out = out.split(gather_sizes.tolist(), dim = 0)
            out = out[rank]

        return out

    return two_chunks.contiguous(), inverse

# zigzag attention
# the context parallelism scheme used to train llama 3 https://arxiv.org/abs/2407.21783

def zig_zag_attn(q, k, v):
    device = q.device
    q = q * (q.shape[-1] ** -0.5)

    # account for grouped query attention

    heads, kv_heads = q.shape[-2], k.shape[-2]
    assert divisible_by(heads, kv_heads)
    q_head_groups = heads // kv_heads

    k, v = tuple(repeat(t, '... h d -> ... (g h) d', g = q_head_groups) for t in (k, v))

    # similarity

    sim = einsum(q, k, 'b i h d, b j h d -> b h i j')

    # masking

    mask_value = -torch.finfo(q.dtype).max
    i, j = sim.shape[-2:]
    causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
    sim = torch.where(causal_mask, mask_value, sim)

    # attend

    attn = sim.softmax(dim = -1)

    # aggregate

    out = einsum(attn, v, 'b h i j, b j h d -> b i h d')

    return out
