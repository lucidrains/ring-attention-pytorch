import math
from functools import partial
from typing import Optional, Tuple
import packaging.version as pkg_version

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.cuda.amp import autocast

from ring_attention_pytorch.ring import (
    ring_pass,
    all_ring_pass,
    null_ring_pass,
    one_ring_pass,
    get_rank,
    get_world_size
)

from beartype import beartype

from einops import rearrange, repeat

# helpers

def exists(v):
    return v is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def first(seq):
    return seq[0]

def pad_at_dim(t, pad: Tuple[int, int], *, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def is_empty(t: Tensor):
    return t.numel() == 0

# make sure triton is installed for forwards

import importlib
from importlib.metadata import version

assert exists(importlib.util.find_spec('triton')), 'latest triton must be installed. `pip install triton -U` first'

triton_version = version('triton')
assert pkg_version.parse(triton_version) >= pkg_version.parse('2.1'), 'triton must be version 2.1 or above. `pip install triton -U` to upgrade'

import triton
import triton.language as tl

from ring_attention_pytorch.triton_flash_attn import (
    flash_attn_backward,
    flash_attn_forward
)

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
        assert all([t.is_cuda for t in (q, k, v)]), 'inputs must be all on cuda'

        dtype = q.dtype
        softmax_scale = q.shape[-1] ** -0.5

        if q.dtype == torch.float32:
            q = q.half()

        if k.dtype == torch.float32:
            k = k.half()

        if v.dtype == torch.float32:
            v = v.half()

        ring_size = default(ring_size, get_world_size())

        cross_attn = q.shape[-3] != k.shape[-3]
        ring_reduce_col &= not cross_attn
        striped_ring_attn &= not cross_attn

        assert k.shape[-1] == v.shape[-1], 'for simplicity when doing ring passing, assume dim_values is equal to dim_queries_keys, majority of transformer do this, not a big issue'

        per_machine_seq_size = k.shape[-3]

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

        orig_k, orig_v, orig_mask, q_seq_len, device = k, v, mask, q.shape[1], q.device

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        kv = torch.stack((k, v))

        # accumulated values

        # o - output
        # m - maximum
        # lse - logsumexp

        o = None
        m = None
        lse = None

        # receive buffers, to be alternated with sent buffer

        receive_kv = None
        receive_mask = None

        # non-causal and causal striped attention can have final normalization of output fused

        can_fuse_final_output_normalization = not causal or (causal and striped_ring_attn)

        for (ring_rank, (is_first, is_last)), ((kv, mask), (receive_kv, receive_mask)) in ring_pass_fn(kv, mask, receive_buffers = (receive_kv, receive_mask), max_iters = max_ring_passes, ring_size = ring_size):
            k, v = kv

            # translate key padding mask to bias

            bias = None

            if exists(mask):
                bias = torch.where(mask, 0.,  float('-inf'))

            # for non-striped attention
            # if the kv ring rank is equal to the current rank (block diagonal), then turn on causal
            # for striped attention, it is always causal, but a lt or gt sign needs to be changed to lte or gte within the cuda code, when determining masking out

            block_causal = False
            causal_mask_diagonal = False

            if causal:
                if striped_ring_attn:
                    block_causal = True
                    causal_mask_diagonal = get_rank() < ring_rank
                else:
                    block_causal = get_rank() == ring_rank

                    if get_rank() < ring_rank:
                        continue

            o, m, lse = flash_attn_forward(
                q, k, v,
                causal = block_causal,
                o = o,
                m = m,
                lse = lse,
                bias = bias,
                softmax_scale = softmax_scale,
                causal_mask_diagonal = causal_mask_diagonal,
                return_normalized_output = can_fuse_final_output_normalization and is_last,
                load_accumulated = not is_first
            )

        if not can_fuse_final_output_normalization:
            m = m[..., :q_seq_len]

            o_scale = torch.exp(m - lse[..., :q_seq_len])
            o.mul_(rearrange(o_scale, 'b h n -> b n h 1'))

        ctx.args = (
            causal,
            softmax_scale,
            orig_mask,
            bucket_size,
            ring_reduce_col,
            max_ring_passes,
            num_lookback_buckets,
            striped_ring_attn,
            ring_size,
            dtype
        )

        ctx.save_for_backward(q, orig_k, orig_v, o, lse)

        # cast back to original dtype

        o = o.type(dtype)
        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 2 in the v2 paper """

        (
            causal,
            softmax_scale,
            mask,
            bucket_size,
            ring_reduce_col,
            max_ring_passes,
            num_lookback_buckets,
            striped_ring_attn,
            ring_size,
            dtype
        ) = ctx.args

        q, k, v, o, lse = ctx.saved_tensors

        do = do.type(o.dtype)

        device = q.device

        if causal:
            mask = None

        row_length = q.shape[-3]

        per_machine_seq_size = k.shape[-3]
        per_machine_buckets = per_machine_seq_size // bucket_size

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        device = q.device

        dq = torch.zeros(q.shape, device = device, dtype = torch.float32)
        dk = torch.zeros(k.shape, device = device, dtype = torch.float32)
        dv = torch.zeros(v.shape, device = device, dtype = torch.float32)

        # k and v will have 16 bits, while dk, dv needs to be kept at 32
        # view everything as int for ring passing
        # for minimizing communication

        k_dtype, v_dtype = k.dtype, v.dtype

        k, v = map(lambda t: t.view(torch.float32), (k, v))
        kv = torch.cat((k, v), dim = -1)

        kv_and_dkv = torch.stack((kv, dk, dv))

        # receive buffers, to be alternated with sent buffer

        receive_kv_and_dkv = None
        receive_mask = None

        # caching the delta (do * o for backwards pass) across ring reduce

        delta = None

        for (ring_rank, _), ((kv_and_dkv, mask), (receive_kv_and_dkv, receive_mask)) in ring_pass_fn(kv_and_dkv, mask, receive_buffers = (receive_kv_and_dkv, receive_mask), max_iters = max_ring_passes, ring_size = ring_size):

            kv, dk, dv = kv_and_dkv

            # reconstitute correct types for k, v, dk, dv

            k, v = kv.chunk(2, dim = -1)
            k, v = k.view(k_dtype), v.view(v_dtype)

            # translate key padding mask to bias

            bias = None

            if exists(mask):
                bias = torch.where(mask, 0., float('-inf'))
                bias = rearrange(bias, 'b j -> b 1 1 j')

            # determine whether to do causal mask or not
            # depends on whether it is striped attention, as well as current machine rank vs ring rank

            if causal and striped_ring_attn:
                need_accum = True
                block_causal = True
                causal_mask_diagonal = get_rank() < ring_rank
            elif causal:
                need_accum = get_rank() >= ring_rank
                block_causal = get_rank() == ring_rank
                causal_mask_diagonal = False
            else:
                need_accum = True
                block_causal = False
                causal_mask_diagonal = False

            # use flash attention backwards kernel to calculate dq, dk, dv and accumulate

            if need_accum:
                ring_dq = torch.empty_like(q)
                ring_dk = torch.empty_like(k)
                ring_dv = torch.empty_like(v)

                with torch.inference_mode():
                    delta = flash_attn_backward(
                        do,
                        q,
                        k,
                        v,
                        o,
                        lse,
                        ring_dq,
                        ring_dk,
                        ring_dv,
                        delta = delta,
                        bias = bias,
                        causal = block_causal,
                        causal_mask_diagonal = causal_mask_diagonal,
                        softmax_scale = softmax_scale
                    )
            else:
                ring_dq, ring_dk, ring_dv = 0., 0., 0.

            dq.add_(ring_dq)
            dk.add_(ring_dk)
            dv.add_(ring_dv)

            if not ring_reduce_col:
                continue

            dkv = kv_and_dkv[1:]

            max_ring_passes = default(max_ring_passes, ring_size)
            dkv = ring_pass(ring_size - max_ring_passes + 1, dkv)

            dk, dv = dkv

        dq, dk, dv = map(lambda t: t.to(dtype), (dq, dk, dv))

        return dq, dk, dv, None, None, None, None, None, None, None

ring_flash_attn_cuda_ = RingFlashAttentionCUDAFunction.apply

@autocast(enabled = False)
@beartype
def ring_flash_attn_cuda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    causal: bool = False,
    bucket_size: int = 1024,
    ring_reduce_col: bool = False,
    striped_ring_attn: bool = False,
    max_lookback_seq_len: Optional[int] = None,
    ring_size: Optional[int] = None
):
    return ring_flash_attn_cuda_(q, k, v, mask, causal, bucket_size, ring_reduce_col, striped_ring_attn, max_lookback_seq_len, ring_size)
