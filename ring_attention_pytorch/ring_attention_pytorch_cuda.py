import math
from functools import partial
from typing import Optional

import torch
from torch import nn, einsum, Tensor
from torch.autograd.function import Function

import einx
from einx import rearrange

from ring_attention_pytorch.ring import (
    ring_pass,
    all_ring_pass,
    null_ring_pass,
    one_ring_pass,
    get_rank,
    get_world_size
)

# make sure flash attention is installed

import importlib

if not exists(importlib.util.find_spec('flash-attn')):
    print('flash-attn must be installed. `pip install flash-attn --no-build-isolation` first')
    exit()

from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward
)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

# ring + (flash) attention forwards and backwards

# flash attention v1 - https://arxiv.org/abs/2205.14135
# flash attention v2 - https://tridao.me/publications/flash2/flash2.pdf
# ring attention - https://arxiv.org/abs/2310.01889

class RingFlashAttentionCUDAFunction(Function):

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        causal: bool,
        bucket_size: int,
        ring_reduce_col: bool,
        striped_ring_attn: bool,
        max_lookback_seq_len: Optional[int],
        ring_size: Optional[int]
    ):
        assert not striped_ring_attn, 'striped ring attention will need to wait for a PR upstream to flash-attn that changes the causal masking for workload balancing'

        ring_size = default(ring_size, get_world_size())

        cross_attn = q.shape[-2] != k.shape[-2]
        ring_reduce_col &= not cross_attn
        striped_ring_attn &= not cross_attn

        assert k.shape[-1] == v.shape[-1]

        per_machine_seq_size = k.shape[-2]

        # calculate max ring passes

        max_ring_passes = None
        num_lookback_buckets = float('inf')

        if exists(max_lookback_seq_len):
            assert causal
            assert not (ring_reduce_col and not divisible_by(per_machine_seq_size, bucket_size))

            max_ring_passes = math.ceil(max_lookback_seq_len / per_machine_seq_size)
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
        all_row_sums = torch.zeros((*q.shape[:-1], 1), device = device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, device = device)

        scale = (q.shape[-1] ** -0.5)

        num_tiles = math.ceil(per_machine_seq_size / bucket_size)

        kv = torch.stack((k, v))

        # receive buffers, to be alternated with sent buffer

        receive_kv = None
        receive_mask = None

        for ring_rank, ((kv, mask), (receive_kv, receive_mask)) in ring_pass_fn(kv, mask, receive_buffers = (receive_kv, receive_mask), max_iters = max_ring_passes, ring_size = ring_size):

            k, v = kv

            col_splits = zip(
                k.split(bucket_size, dim = -2),
                v.split(bucket_size, dim = -2),
                maybe_split(mask, bucket_size, dim = -1)
            )

            for k_ind, (kc, vc, col_mask) in enumerate(col_splits):
                col_ring_rank = ring_rank % ring_size
                col_bucket_index = col_ring_rank * per_machine_buckets + k_ind

                row_splits = zip(
                    q.split(bucket_size, dim = -2),
                    o.split(bucket_size, dim = -2),
                    all_row_sums.split(bucket_size, dim = -2),
                    all_row_maxes.split(bucket_size, dim = -2),
                )

                for ind, (qc, oc, row_sums, row_maxes) in enumerate(row_splits):

                    qk_len_diff = kc.shape[-2] - qc.shape[-2]

                    row_bucket_index = row_ring_rank * per_machine_buckets + ind

                    attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                    if exists(col_mask):
                        attn_weights = einx.where('b j, b h i j, -> b h i j', col_mask, attn_weights, max_neg_value)

                    if causal:
                        if (row_bucket_index - col_bucket_index) > num_lookback_buckets:
                            continue

                        if striped_ring_attn:
                            # `GetMaskStripedAttention` pseudocode at end of section 2.2.1 of https://arxiv.org/abs/2311.09431

                            triu_offset = int(row_bucket_index >= col_bucket_index)
                            causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(triu_offset + qk_len_diff)
                            attn_weights.masked_fill_(causal_mask, max_neg_value)

                        else:
                            if row_bucket_index == col_bucket_index:
                                causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(1 + qk_len_diff)
                                attn_weights.masked_fill_(causal_mask, max_neg_value)
                            elif row_bucket_index < col_bucket_index:
                                attn_weights.fill_(max_neg_value)

                    block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                    new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                    exp_weights = torch.exp(attn_weights - new_row_maxes)

                    if exists(col_mask):
                        exp_weights = einx.where('b j, b h i j, -> b h i j', col_mask, exp_weights, 0.)

                    block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = EPSILON)

                    exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

                    exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)

                    new_row_sums = exp_row_max_diff * row_sums + block_row_sums

                    oc.mul_(exp_row_max_diff).add_(exp_values)

                    row_maxes.copy_(new_row_maxes)
                    row_sums.copy_(new_row_sums)

        o.div_(all_row_sums)

        lse = all_row_sums.log() + all_row_maxes

        ctx.args = (
            causal,
            scale,
            orig_mask,
            bucket_size,
            ring_reduce_col,
            max_ring_passes,
            num_lookback_buckets,
            striped_ring_attn,
            ring_size
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
            ring_size
        ) = ctx.args

        q, k, v, o, lse = ctx.saved_tensors

        row_ring_rank = (get_rank() % ring_size) if ring_reduce_col else 0

        per_machine_seq_size = k.shape[-2]
        per_machine_buckets = per_machine_seq_size // bucket_size

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        kv_and_dkv = torch.stack((k, v, dk, dv))

        # receive buffers, to be alternated with sent buffer

        receive_kv_and_dkv = None
        receive_mask = None

        for ring_rank, ((kv_and_dkv, mask), (receive_kv_and_dkv, receive_mask)) in ring_pass_fn(kv_and_dkv, mask, receive_buffers = (receive_kv_and_dkv, receive_mask), max_iters = max_ring_passes, ring_size = ring_size):

            k, v, dk, dv = kv_and_dkv

            ring_dq, ring_dk, ring_dv, *_ = _flash_attn_varlen_backward(
                dout = do,
                q = q,
                k = k,
                v = v,
                out = o,
                softmax_lse = lse,
                dq = torch.zeros_like(dq),
                dk = torch.zeros_like(dk),
                dv = torch.zeros_like(dv),
                cu_seqlens_q = q.shape[-2],
                cu_seqlens_k = k.shape[-2],
                max_seqlen_q = None,
                max_seqlen_k = None,
                dropout_p = 0.,
                softmax_scale = scale,
                causal = causal,
                window_size = bucket_size,
                alibi_slopes = None,
                deterministic = False
            )

            dq.add_(ring_dq)
            dk.add_(ring_dk)
            dv.add_(ring_dv)

            if not ring_reduce_col:
                continue

            dkv = kv_and_dkv[2:]

            max_ring_passes = default(max_ring_passes, ring_size)
            dkv = ring_pass(ring_size - max_ring_passes + 1, dkv)

            dk, dv = dkv

        return dq, dk, dv, None, None, None, None, None, None, None

ring_flash_attn_cuda = RingFlashAttentionCUDAFunction.apply
