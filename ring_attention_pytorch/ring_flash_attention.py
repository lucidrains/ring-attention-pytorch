import math
from functools import partial
from typing import Optional

import torch
from torch import nn, einsum, Tensor
from torch.autograd.function import Function

import einx
from einx import rearrange

from ring_attention_pytorch.ring import (
    all_ring_pass,
    null_ring_pass,
    one_ring_pass,
    get_rank
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

    return t.chunk(size, dim = dim)

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
        q_bucket_size: int,
        k_bucket_size: int,
        ring_reduce_col: bool
    ):
        """ Algorithm 1 in the v2 paper """

        orig_k, orig_v, device = k, v, q.device

        row_ring_rank = get_rank() if ring_reduce_col else 0
        per_machine_row_size = q.shape[-2]
        per_machine_col_size = k.shape[-2]

        rank_row_offset = row_ring_rank * per_machine_row_size

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), device = device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, device = device)

        scale = (q.shape[-1] ** -0.5)

        num_row_tiles = math.ceil(q.shape[-2] / q_bucket_size)
        num_col_tiles = math.ceil(k.shape[-2] / k_bucket_size)

        if exists(mask):
            if mask.ndim == 2:
                mask = rearrange('b n -> b 1 1 n', mask)

            mask = ((mask,) * num_row_tiles) if mask.shape[-2] == 1 else mask.split(q_bucket_size, dim = -2)
        else:
            mask = (None,) * num_row_tiles

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            mask,
            all_row_sums.split(q_bucket_size, dim = -2),
            all_row_maxes.split(q_bucket_size, dim = -2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size + rank_row_offset - qk_len_diff

            for ring_rank, (k, v, row_mask) in ring_pass_fn(k, v, row_mask):

                per_machine_col_offset = ring_rank * per_machine_col_size

                col_splits = zip(
                    k.split(k_bucket_size, dim = -2),
                    v.split(k_bucket_size, dim = -2),
                    maybe_split(row_mask, k_bucket_size, dim = -1)
                )

                for k_ind, (kc, vc, col_mask) in enumerate(col_splits):
                    k_start_index = k_ind * k_bucket_size + per_machine_col_offset

                    attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                    if exists(col_mask):
                        attn_weights.masked_fill_(~col_mask, max_neg_value)

                    if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                        causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                        attn_weights.masked_fill_(causal_mask, max_neg_value)

                    block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                    new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                    exp_weights = torch.exp(attn_weights - new_row_maxes)

                    if exists(col_mask):
                        exp_weights.masked_fill_(~col_mask, 0.)

                    block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = EPSILON)

                    exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

                    exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)

                    new_row_sums = exp_row_max_diff * row_sums + block_row_sums

                    oc.mul_(exp_row_max_diff).add_(exp_values)

                    row_maxes.copy_(new_row_maxes)
                    row_sums.copy_(new_row_sums)

            oc.div_(row_sums)

        lse = all_row_sums.log() + all_row_maxes

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size, ring_reduce_col)
        ctx.save_for_backward(q, orig_k, orig_v, o, lse)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 2 in the v2 paper """

        causal, scale, mask, q_bucket_size, k_bucket_size, ring_reduce_col = ctx.args
        q, k, v, o, lse = ctx.saved_tensors

        row_ring_rank = get_rank() if ring_reduce_col else 0
        per_machine_row_size = q.shape[-2]
        per_machine_col_size = k.shape[-2]

        rank_row_offset = row_ring_rank * per_machine_row_size

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            do.split(q_bucket_size, dim = -2),
            mask,
            lse.split(q_bucket_size, dim = -2),
            dq.split(q_bucket_size, dim = -2)
        )

        for ind, (qc, oc, doc, row_mask, lsec, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size + rank_row_offset - qk_len_diff

            for ring_rank, (k, v, row_mask, dk, dv) in ring_pass_fn(k, v, row_mask, dk, dv):

                per_machine_col_offset = ring_rank * per_machine_col_size

                col_splits = zip(
                    k.split(k_bucket_size, dim = -2),
                    v.split(k_bucket_size, dim = -2),
                    dk.split(k_bucket_size, dim = -2),
                    dv.split(k_bucket_size, dim = -2),
                    maybe_split(row_mask, k_bucket_size, dim = -1)
                )

                for k_ind, (kc, vc, dkc, dvc, col_mask) in enumerate(col_splits):
                    k_start_index = k_ind * k_bucket_size + per_machine_col_offset

                    attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                    if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                        causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                        attn_weights.masked_fill_(causal_mask, max_neg_value)

                    p = torch.exp(attn_weights - lsec)

                    if exists(col_mask):
                        p.masked_fill_(~col_mask, 0.)

                    dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                    dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                    D = (doc * oc).sum(dim = -1, keepdims = True)
                    ds = p * scale * (dp - D)

                    dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                    dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                    dqc.add_(dq_chunk)
                    dkc.add_(dk_chunk)
                    dvc.add_(dv_chunk)

            dk = one_ring_pass(dk)
            dv = one_ring_pass(dv)

        return dq, dk, dv, None, None, None, None, None

ring_flash_attn = RingFlashAttentionFunction.apply
