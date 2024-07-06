from __future__ import annotations

from math import ceil
from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.autograd.function import Function

from einops import rearrange, repeat, reduce

from ring_attention_pytorch.ring import (
    ring_pass,
    all_ring_pass,
    null_ring_pass,
    one_ring_pass,
    get_rank,
    get_world_size
)

from beartype import beartype

# constants

EPSILON = 1e-10

# helper functions

def exists(val):
    return val is not None

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def none_iterator():
    while True:
        yield None

def maybe_split(t, size, dim = -2):
    if not exists(t):
        return none_iterator()

    return t.split(size, dim = dim)

def softclamp(t, value):
    return (t / value).tanh() * value

# ring + (flash) attention forwards and backwards

# flash attention v1 - https://arxiv.org/abs/2205.14135
# flash attention v2 - https://tridao.me/publications/flash2/flash2.pdf
# ring attention - https://arxiv.org/abs/2310.01889

class RingFlashAttentionFunction(Function):

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None,
        causal: bool,
        bucket_size: int,
        ring_reduce_col: bool,
        striped_ring_attn: bool,
        max_lookback_seq_len: int | None,
        ring_size: int | None,
        softclamp_qk_sim: bool,
        softclamp_value: float
    ):
        ring_size = default(ring_size, get_world_size())

        cross_attn = q.shape[-3] != k.shape[-3]
        ring_reduce_col &= not cross_attn
        striped_ring_attn &= not cross_attn

        assert k.shape[-2:] == v.shape[-2:]
        q_heads, kv_heads = q.shape[-2], k.shape[-2]

        assert divisible_by(q_heads, kv_heads)
        q_head_groups = q_heads // kv_heads

        per_machine_seq_size = k.shape[1]

        # calculate max ring passes

        max_ring_passes = None
        num_lookback_buckets = float('inf')

        if exists(max_lookback_seq_len):
            assert causal
            assert not (ring_reduce_col and not divisible_by(per_machine_seq_size, bucket_size))

            max_ring_passes = ceil(max_lookback_seq_len / per_machine_seq_size)
            num_lookback_buckets = max_lookback_seq_len // bucket_size

        # ignore key padding mask if autoregressive

        if causal:
            mask = None

        bucket_size = min(per_machine_seq_size, bucket_size)
        per_machine_buckets = per_machine_seq_size // bucket_size

        orig_k, orig_v, orig_mask, device = k, v, mask, q.device

        row_ring_rank = (get_rank() % ring_size) if ring_reduce_col else 0

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        max_neg_value = -torch.finfo(q.dtype).max

        o = torch.zeros_like(q)
        batch, seq, heads, _ = q.shape

        all_row_sums = torch.zeros((batch, heads, seq, 1), device = device)
        all_row_maxes = torch.full((batch, heads, seq, 1), max_neg_value, device = device)

        scale = (q.shape[-1] ** -0.5)

        kv = torch.stack((k, v))

        # receive buffers, to be alternated with sent buffer

        receive_kv = None
        receive_mask = None

        for (ring_rank, _), ((kv, mask), (receive_kv, receive_mask)) in ring_pass_fn(kv, mask, receive_buffers = (receive_kv, receive_mask), max_iters = max_ring_passes, ring_size = ring_size):

            k, v = kv

            # account for grouped query attention

            k, v = map(lambda t: repeat(t, '... h d -> ... (g h) d', g = q_head_groups), (k, v))

            col_splits = zip(
                k.split(bucket_size, dim = -3),
                v.split(bucket_size, dim = -3),
                maybe_split(mask, bucket_size, dim = -1)
            )

            for k_ind, (kc, vc, col_mask) in enumerate(col_splits):
                col_ring_rank = ring_rank % ring_size
                col_bucket_index = col_ring_rank * per_machine_buckets + k_ind

                row_splits = zip(
                    q.split(bucket_size, dim = -3),
                    o.split(bucket_size, dim = -3),
                    all_row_sums.split(bucket_size, dim = -2),
                    all_row_maxes.split(bucket_size, dim = -2),
                )

                for ind, (qc, oc, row_sums, row_maxes) in enumerate(row_splits):

                    row_bucket_index = row_ring_rank * per_machine_buckets + ind

                    attn_weights = einsum('b i h d, b j h d -> b h i j', qc, kc) * scale

                    if softclamp_qk_sim:
                        attn_weights = softclamp(attn_weights, softclamp_value)

                    if exists(col_mask):
                        col_mask_unsqueezed = rearrange(col_mask, 'b j -> b 1 1 j')
                        attn_weights = attn_weights.masked_fill(~col_mask_unsqueezed, max_neg_value)

                    if causal:
                        qk_len_diff = kc.shape[-3] - qc.shape[-3]

                        if (row_bucket_index - col_bucket_index) > num_lookback_buckets:
                            continue

                        if striped_ring_attn:
                            # `GetMaskStripedAttention` pseudocode at end of section 2.2.1 of https://arxiv.org/abs/2311.09431

                            triu_offset = int(row_bucket_index >= col_bucket_index)
                            causal_mask = torch.ones((qc.shape[-3], kc.shape[-3]), dtype = torch.bool, device = device).triu(triu_offset + qk_len_diff)
                            attn_weights.masked_fill_(causal_mask, max_neg_value)

                        else:
                            if row_bucket_index == col_bucket_index:
                                causal_mask = torch.ones((qc.shape[-3], kc.shape[-3]), dtype = torch.bool, device = device).triu(1 + qk_len_diff)
                                attn_weights.masked_fill_(causal_mask, max_neg_value)
                            elif row_bucket_index < col_bucket_index:
                                attn_weights.fill_(max_neg_value)

                    block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                    new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                    exp_weights = torch.exp(attn_weights - new_row_maxes)

                    if exists(col_mask):
                        exp_weights = exp_weights.masked_fill(~col_mask_unsqueezed, 0.)

                    block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = EPSILON)

                    exp_values = einsum('b h i j, b j h d -> b i h d', exp_weights, vc)

                    exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)

                    new_row_sums = exp_row_max_diff * row_sums + block_row_sums

                    exp_row_max_diff = rearrange(exp_row_max_diff, 'b h n 1 -> b n h 1')
                    oc.mul_(exp_row_max_diff).add_(exp_values)

                    row_maxes.copy_(new_row_maxes)
                    row_sums.copy_(new_row_sums)

        o.div_(rearrange(all_row_sums, 'b h n 1 -> b n h 1'))

        lse = all_row_sums.clamp(min = EPSILON).log() + all_row_maxes

        ctx.args = (
            causal,
            scale,
            orig_mask,
            bucket_size,
            ring_reduce_col,
            max_ring_passes,
            num_lookback_buckets,
            striped_ring_attn,
            ring_size,
            q_head_groups,
            softclamp_qk_sim,
            softclamp_value
        )

        ctx.save_for_backward(q, orig_k, orig_v, o, lse)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 2 in the v2 paper """

        (
            causal,
            scale,
            mask,
            bucket_size,
            ring_reduce_col,
            max_ring_passes,
            num_lookback_buckets,
            striped_ring_attn,
            ring_size,
            q_head_groups,
            softclamp_qk_sim,
            softclamp_value
        ) = ctx.args

        q, k, v, o, lse = ctx.saved_tensors

        row_ring_rank = (get_rank() % ring_size) if ring_reduce_col else 0

        per_machine_seq_size = k.shape[1]
        per_machine_buckets = per_machine_seq_size // bucket_size

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # kv and dkv sent around the ring in one go

        kv_and_dkv = torch.stack((k, v, dk, dv))

        # receive buffers, to be alternated with sent buffer

        receive_kv_and_dkv = None
        receive_mask = None

        # precompute the softmax D

        D = (do * o).sum(dim = -1, keepdims = True)
        D = rearrange(D, 'b n h 1 -> b h n 1')

        # ring reduce key / values

        for (ring_rank, _), ((kv_and_dkv, mask), (receive_kv_and_dkv, receive_mask)) in ring_pass_fn(kv_and_dkv, mask, receive_buffers = (receive_kv_and_dkv, receive_mask), max_iters = max_ring_passes, ring_size = ring_size):
            k_ring_rank = ring_rank % ring_size

            k, v, dk, dv = kv_and_dkv

            # account for grouped query attention

            k, v = map(lambda t: repeat(t, '... h d -> ... (g h) d', g = q_head_groups), (k, v))

            col_splits = zip(
                k.split(bucket_size, dim = 1),
                v.split(bucket_size, dim = 1),
                dk.split(bucket_size, dim = 1),
                dv.split(bucket_size, dim = 1),
                maybe_split(mask, bucket_size, dim = -1)
            )

            for k_ind, (kc, vc, dkc, dvc, col_mask) in enumerate(col_splits):
                col_bucket_index = k_ring_rank * per_machine_buckets + k_ind

                row_splits = zip(
                    q.split(bucket_size, dim = 1),
                    do.split(bucket_size, dim = 1),
                    D.split(bucket_size, dim = -2),
                    lse.split(bucket_size, dim = -2),
                    dq.split(bucket_size, dim = 1)
                )

                for ind, (qc, doc, Dc, lsec, dqc) in enumerate(row_splits):
                    row_bucket_index = row_ring_rank * per_machine_buckets + ind

                    attn_weights = einsum('b i h d, b j h d -> b h i j', qc, kc) * scale

                    if softclamp_qk_sim:
                        attn_weights = softclamp(attn_weights, softclamp_value)

                    if causal:
                        if (row_bucket_index - col_bucket_index) > num_lookback_buckets:
                            continue

                        if striped_ring_attn:
                            # `GetMaskStripedAttention` pseudocode at end of section 2.2.1 of https://arxiv.org/abs/2311.09431

                            triu_offset = int(row_bucket_index >= col_bucket_index)
                            causal_mask = torch.ones((qc.shape[1], kc.shape[1]), dtype = torch.bool, device = device).triu(triu_offset)
                            attn_weights.masked_fill_(causal_mask, max_neg_value)
                        else:
                            if row_bucket_index == col_bucket_index:
                                causal_mask = torch.ones((qc.shape[1], kc.shape[1]), dtype = torch.bool, device = device).triu(1)
                                attn_weights.masked_fill_(causal_mask, max_neg_value)
                            elif row_bucket_index < col_bucket_index:
                                attn_weights.fill_(max_neg_value)

                    p = torch.exp(attn_weights - lsec)

                    if exists(col_mask):
                        col_mask_unsqueezed = rearrange(col_mask, 'b j -> b 1 1 j')
                        p = p.masked_fill(~col_mask_unsqueezed, 0.)

                    dv_chunk = einsum('b h i j, b i h d -> b j h d', p, doc)
                    dp = einsum('b i h d, b j h d -> b h i j', doc, vc)

                    ds = p * scale * (dp - Dc)

                    # backwards of tanh(x) for softclamping is (1. - tanh(x)^2)

                    if softclamp_qk_sim:
                        attn_weights = (log(p) + lsec) / softclamp_value
                        ds *= (1. - attn_weights ** 2)

                    # dq and dk chunks

                    dq_chunk = einsum('b h i j, b j h d -> b i h d', ds, kc)
                    dk_chunk = einsum('b h i j, b i h d -> b j h d', ds, qc)

                    # account for grouped query attention

                    dk_chunk = reduce(dk_chunk, '... (g h) d -> ... h d', g = q_head_groups, reduction = 'sum')
                    dv_chunk = reduce(dv_chunk, '... (g h) d -> ... h d', g = q_head_groups, reduction = 'sum')

                    dqc.add_(dq_chunk)
                    dkc.add_(dk_chunk)
                    dvc.add_(dv_chunk)

            if not ring_reduce_col:
                continue

            dkv = kv_and_dkv[2:]

            max_ring_passes = default(max_ring_passes, ring_size)
            dkv = ring_pass(ring_size - max_ring_passes + 1, dkv)

            dk, dv = dkv

        return dq, dk, dv, None, None, None, None, None, None, None, None, None

ring_flash_attn_ = RingFlashAttentionFunction.apply

@beartype
def ring_flash_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    causal: bool = False,
    bucket_size: int = 1024,
    ring_reduce_col: bool = False,
    striped_ring_attn: bool = False,
    max_lookback_seq_len: int | None = None,
    ring_size: int | None = None,
    softclamp_qk_sim: bool = False,
    softclamp_value: float = 50.
):
    return ring_flash_attn_(q, k, v, mask, causal, bucket_size, ring_reduce_col, striped_ring_attn, max_lookback_seq_len, ring_size, softclamp_qk_sim, softclamp_value)
