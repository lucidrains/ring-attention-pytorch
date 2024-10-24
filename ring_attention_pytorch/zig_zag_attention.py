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

from ring_attention_pytorch.tensor_typing import Float

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
        t, gather_sizes = AllGather(dim = 0)(t)

    chunks = 2 * world_size
    t = t.chunk(chunks, dim = -2)

    # each rank takes care of two chunks for their simple workload balancing scheme

    two_chunks = torch.cat((t[rank], t[chunks - 1 - rank]), dim = -2)

    # inverse

    def inverse(two_chunks):

        two_chunks = rearrange(two_chunks, 'b (two c) d -> b two c d', two = 2)
        all_chunks, _ = AllGather(dim = -3)(two_chunks)

        first_half, second_half = rearrange(all_chunks, 'b (all_pairs two) c d -> two b all_pairs c d', two = 2)

        out = torch.cat((first_half, second_half.flip(dims = (-3,))), dim = -3)
        out = rearrange(out, 'b n c d -> b (n c) d')

        if all_gather_batch:
            out = out.split(gather_sizes.tolist(), dim = 0)
            out = out[rank]

        return out

    return two_chunks.contiguous(), inverse

# zigzag attention
# the context parallelism scheme used to train llama 3 https://arxiv.org/abs/2407.21783

def zig_zag_attn(
    q: Float['b h n dq'],
    k: Float['b h j dq'],
    v: Float['b h j dv']
) -> Float['b h j dv']:

    twice_chunk_size, device = q.shape[-2], q.device
    q = q * (q.shape[-1] ** -0.5)

    # account for grouped query attention

    heads, kv_heads = q.shape[1], k.shape[1]
    assert divisible_by(heads, kv_heads)
    q_head_groups = heads // kv_heads

    # keys and values are all gathered, only works for grouped query attention

    all_gather_seq = AllGather(dim = -2)

    k, _ = all_gather_seq(k)
    v, _ = all_gather_seq(v)

    # expand the keys and values to match all query attention heads

    k, v = tuple(repeat(t, 'b h n d -> b (g h) n d', g = q_head_groups) for t in (k, v))

    # similarity

    sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

    # masking
    # todo - handle specialized masking, and leverage flex attention if cuda

    chunk_size = twice_chunk_size // 2
    world_size, rank = get_world_size(), get_rank()

    mask_value = -torch.finfo(q.dtype).max
    seq_len = sim.shape[-1]

    full_causal_mask = torch.ones((seq_len, seq_len), dtype = torch.bool, device = device).triu(1)
    forward_causal_mask, reverse_causal_mask = rearrange(full_causal_mask, '(i c1) (two j c2) -> two i c1 j c2', c1 = chunk_size, c2 = chunk_size, two = 2)

    chunked_causal_mask = rearrange([
        forward_causal_mask,
        reverse_causal_mask.flip(dims = (-2,))
    ] ,'two i c1 j c2 -> i c1 (j two c2)')

    causal_mask = torch.cat((
        chunked_causal_mask[rank],
        chunked_causal_mask[(2 * world_size) - 1 - rank]
    ), dim = -2)

    sim = torch.where(causal_mask, mask_value, sim)

    # attend

    attn = sim.softmax(dim = -1)

    # aggregate

    out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

    return out
