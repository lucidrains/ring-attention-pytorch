import math
from functools import partial
from typing import Optional

import torch
from torch import nn, einsum, Tensor
from torch.autograd.function import Function

import einx
from einx import rearrange

from ring_attention_pytorch.ring import (
    maybe,
    ring_pass,
    all_ring_pass,
    null_ring_pass,
    one_ring_pass,
    get_rank,
    get_world_size
)

# constants

EPSILON = 1e-10

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def none_iterator():
    while True:
        yield None

def maybe_split(t, size, dim = -2):
    if not exists(t):
        return none_iterator()

    return t.split(size, dim = dim)

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
        mask: Optional[Tensor],
        causal: bool,
        bucket_size: int,
        ring_reduce_col: bool,
        max_ring_passes: Optional[int],
        striped_ring_attn: bool,
    ):
        assert q.shape[-2] == k.shape[-2]
        assert k.shape[-1] == v.shape[-1]

        """ Algorithm 1 in the flash attention v2 paper + ring passing """

        # ignore key padding mask if autoregressive

        if causal:
            mask = None

        per_machine_seq_size = q.shape[-2]
        bucket_size = min(per_machine_seq_size, bucket_size)
        per_machine_buckets = per_machine_seq_size // bucket_size

        orig_k, orig_v, orig_mask, device = k, v, mask, q.device

        row_ring_rank = get_rank() if ring_reduce_col else 0

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        max_neg_value = -torch.finfo(q.dtype).max

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), device = device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, device = device)

        scale = (q.shape[-1] ** -0.5)

        num_tiles = math.ceil(per_machine_seq_size / bucket_size)

        kv = torch.stack((k, v))

        for ring_rank, (kv, mask) in ring_pass_fn(kv, mask, max_iters = max_ring_passes):

            k, v = kv

            col_splits = zip(
                k.split(bucket_size, dim = -2),
                v.split(bucket_size, dim = -2),
                maybe_split(mask, bucket_size, dim = -1)
            )

            for k_ind, (kc, vc, col_mask) in enumerate(col_splits):
                col_bucket_index = ring_rank * per_machine_buckets + k_ind

                row_splits = zip(
                    q.split(bucket_size, dim = -2),
                    o.split(bucket_size, dim = -2),
                    all_row_sums.split(bucket_size, dim = -2),
                    all_row_maxes.split(bucket_size, dim = -2),
                )

                for ind, (qc, oc, row_sums, row_maxes) in enumerate(row_splits):
                    row_bucket_index = row_ring_rank * per_machine_buckets + ind

                    attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                    if exists(col_mask):
                        attn_weights = einx.where('b j, b h i j, -> b h i j', col_mask, attn_weights, max_neg_value)

                    if causal:
                        if striped_ring_attn:
                            # `GetMaskStripedAttention` pseudocode at end of section 2.2.1 of https://arxiv.org/abs/2311.09431

                            triu_offset = int(row_bucket_index >= col_bucket_index)
                            causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(triu_offset)
                            attn_weights.masked_fill_(causal_mask, max_neg_value)

                        else:
                            if row_bucket_index == col_bucket_index:
                                causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(1)
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
            striped_ring_attn
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
            striped_ring_attn
        ) = ctx.args

        q, k, v, o, lse = ctx.saved_tensors

        row_ring_rank = get_rank() if ring_reduce_col else 0

        per_machine_seq_size = q.shape[-2]
        per_machine_buckets = per_machine_seq_size // bucket_size

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(bucket_size, dim = -2),
            o.split(bucket_size, dim = -2),
            do.split(bucket_size, dim = -2),
            lse.split(bucket_size, dim = -2),
            dq.split(bucket_size, dim = -2)
        )

        kv_and_dkv = torch.stack((k, v, dk, dv))

        for ring_rank, (kv_and_dkv, mask) in ring_pass_fn(kv_and_dkv, mask, max_iters = max_ring_passes):

            k, v, dk, dv = kv_and_dkv

            col_splits = zip(
                k.split(bucket_size, dim = -2),
                v.split(bucket_size, dim = -2),
                dk.split(bucket_size, dim = -2),
                dv.split(bucket_size, dim = -2),
                maybe_split(mask, bucket_size, dim = -1)
            )

            for k_ind, (kc, vc, dkc, dvc, col_mask) in enumerate(col_splits):
                col_bucket_index = ring_rank * per_machine_buckets + k_ind

                for ind, (qc, oc, doc, lsec, dqc) in enumerate(row_splits):
                    row_bucket_index = row_ring_rank * per_machine_buckets + ind

                    attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                    if causal:
                        if striped_ring_attn:
                            # `GetMaskStripedAttention` pseudocode at end of section 2.2.1 of https://arxiv.org/abs/2311.09431

                            triu_offset = int(row_bucket_index >= col_bucket_index)
                            causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(triu_offset)
                            attn_weights.masked_fill_(causal_mask, max_neg_value)
                        else:
                            if row_bucket_index == col_bucket_index:
                                causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(1)
                                attn_weights.masked_fill_(causal_mask, max_neg_value)
                            elif row_bucket_index < col_bucket_index:
                                attn_weights.fill_(max_neg_value)

                    p = torch.exp(attn_weights - lsec)

                    if exists(col_mask):
                        p = einx.where('b j, b h i j, -> b h i j', col_mask, p, 0.)

                    dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                    dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                    D = (doc * oc).sum(dim = -1, keepdims = True)
                    ds = p * scale * (dp - D)

                    dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                    dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                    dqc.add_(dq_chunk)
                    dkc.add_(dk_chunk)
                    dvc.add_(dv_chunk)

            dkv = kv_and_dkv[2:]

            max_ring_passes = default(max_ring_passes, get_world_size())
            dkv = ring_pass(get_world_size() - max_ring_passes + 1, dkv)

            dk, dv = dkv

        return dq, dk, dv, None, None, None, None, None, None, None

ring_flash_attn = RingFlashAttentionFunction.apply
