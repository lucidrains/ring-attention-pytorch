from math import ceil
from collections import namedtuple

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

ShardOutput = namedtuple('ShardOutput', [
    'local_sequence',
    'query_positions',
    'key_value_positions'
])

def zig_zag_shard(t, all_gather_batch = False):
    device, seq_len = t.device, t.shape[-2]
    rank, world_size = get_rank(), get_world_size()

    if all_gather_batch:
        t, gather_sizes = AllGather(dim = 0)(t)

    chunks = 2 * world_size
    chunk_size = seq_len // chunks

    t = t.chunk(chunks, dim = -2)

    # each rank takes care of two chunks for their simple workload balancing scheme

    two_chunks = torch.cat((t[rank], t[chunks - 1 - rank]), dim = -2).contiguous()

    # take care of positions

    pos = torch.arange(seq_len, device = device)

    pos = rearrange(pos, '(n chunk_size) -> n chunk_size', chunk_size = chunk_size)
    pos_first_half, pos_second_half = pos.chunk(2, dim = -2)
    pos = torch.stack((pos_first_half, pos_second_half.flip(dims = (-2,))), dim = -2)

    q_indices = rearrange(pos[rank], '... -> (...)')
    kv_indices = rearrange(pos, '... -> (...)')

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

    return ShardOutput(two_chunks, q_indices, kv_indices), inverse

# zigzag attention
# the context parallelism scheme used to train llama 3 https://arxiv.org/abs/2407.21783

def zig_zag_attn(
    q: Float['b qh i dq'],
    k: Float['b h j dq'],
    v: Float['b h j dv'],
    dropout = 0.,
    attn_mask = None
) -> Float['b qh i dv']:

    twice_chunk_size, device = q.shape[-2], q.device

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

    out = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p = dropout,
        attn_mask = attn_mask
    )

    return out
