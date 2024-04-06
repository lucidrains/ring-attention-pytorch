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

def is_contiguous(x: Tensor):
    return x.stride(-1) == 1

def padded_false_on_right_side(t: Tensor):
    if t.shape[-1] <= 1:
        return True

    false_to_true = ~t[..., :-1] & t[..., 1:]
    return not false_to_true.any()

# make sure flash attention is installed for backwards

import importlib
from importlib.metadata import version

assert exists(importlib.util.find_spec('flash_attn')), 'flash-attn must be installed. `pip install flash-attn --no-build-isolation` first'

flash_attn_version = version('flash_attn')
assert pkg_version.parse(flash_attn_version) >= pkg_version.parse('2.5.1')

from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_backward
)

from flash_attn.bert_padding import (
    pad_input,
    unpad_input
)

@beartype
def unpad_inputs_and_return_inverse_fn(
    tensors: Tuple[Tensor, ...],
    mask: Tensor
):
    assert len(tensors) > 0
    batch, seqlen, *_ = first(tensors).shape

    outs = []

    for tensor in tensors:
        out, indices, cu_seqlens, max_seqlen = unpad_input(tensor, mask)
        outs.append(out)

    def inverse_fn(y):
        return pad_input(y, indices, batch, seqlen)

    return tuple(outs), cu_seqlens, max_seqlen, inverse_fn

# make sure triton is installed for forwards

assert exists(importlib.util.find_spec('triton')), 'latest triton must be installed. `pip install triton -U` first'

triton_version = version('triton')
assert pkg_version.parse(triton_version) >= pkg_version.parse('2.1')

import triton
import triton.language as tl

# taking the flash attention forwards from Tri's flash_attn repository
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
# and modifying to return unnormalized accumulation, row maxes, row lse - reduced over passed rings

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    M,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    HAS_BIAS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    CAUSAL_MASK_DIAGONAL: tl.constexpr,
    LOAD_ACCUMULATED: tl.constexpr,
    RETURN_NORMALIZED_OUTPUT: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    if HAS_BIAS:
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n

    # maximum

    m_ptrs = M + off_hb * seqlen_q_rounded + offs_m

    if LOAD_ACCUMULATED:
        m_i = tl.load(m_ptrs)
    else:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # load lse

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m

    if LOAD_ACCUMULATED:
        lse_i = tl.load(lse_ptrs)
    else:
        lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # load accumualted output

    offs_d = tl.arange(0, BLOCK_HEADDIM)

    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )

    if LOAD_ACCUMULATED:
        if EVEN_M:
            if EVEN_HEADDIM:
                acc_o = tl.load(out_ptrs)
            else:
                acc_o = tl.load(out_ptrs, mask=offs_d[None, :] < headdim)
        else:
            if EVEN_HEADDIM:
                acc_o = tl.load(out_ptrs, mask=offs_m[:, None] < seqlen_q)
            else:
                acc_o = tl.load(
                    out_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
                )

        acc_o = acc_o.to(tl.float32)
    else:
        acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # load queries, keys, values

    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )

    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        if IS_CAUSAL:
            if CAUSAL_MASK_DIAGONAL:
                # needed for stripe attention
                qk += tl.where(offs_m[:, None] > (start_n + offs_n)[None, :], 0, float("-inf"))
            else:
                qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        if HAS_BIAS:
            if EVEN_N:
                bias = tl.load(b_ptrs + start_n)
            else:
                bias = tl.load(
                    b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                )
            bias = bias[None, :]

            bias = bias.to(tl.float32)
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])

        l_ij = tl.sum(p, 1)

        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics

        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    if RETURN_NORMALIZED_OUTPUT:
        acc_o_scale = tl.exp(m_i - lse_i)
        acc_o = acc_o * acc_o_scale[:, None]

    # offsets for m and lse

    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # write back lse and m

    tl.store(lse_ptrs, lse_i)

    if not RETURN_NORMALIZED_OUTPUT:
        tl.store(m_ptrs, m_i)

    # write to output

    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )

def flash_attn_forward(
    q,
    k,
    v,
    bias = None,
    causal = False,
    o = None,
    m = None,
    lse = None,
    softmax_scale = None,
    causal_mask_diagonal = False,
    return_normalized_output = False,
    load_accumulated = True
):
    q, k, v = [x if is_contiguous(x) else x.contiguous() for x in (q, k, v)]

    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape

    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda

    softmax_scale = default(softmax_scale, d ** -0.5)

    has_bias = exists(bias)

    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda

        if bias.ndim == 2:
            bias = repeat(bias, 'b j -> b h i j', h = nheads, i = seqlen_q)

        if not is_contiguous(bias):
            bias = bias.contiguous()

        assert bias.shape[-2:] == (seqlen_q, seqlen_k)
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)

    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128

    if not exists(lse):
        max_neg_value = -torch.finfo(torch.float32).max
        init_fn = partial(torch.full, fill_value = max_neg_value) if load_accumulated else torch.empty
        lse = init_fn((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    if not exists(m):
        max_neg_value = -torch.finfo(torch.float32).max
        init_fn = partial(torch.full, fill_value = max_neg_value) if load_accumulated else torch.empty
        m = init_fn((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    if not exists(o):
        init_fn = torch.zeros_like if load_accumulated else torch.empty_like
        o = init_fn(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        m,
        lse,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,
        has_bias,
        causal,
        causal_mask_diagonal,
        load_accumulated,
        return_normalized_output,
        BLOCK_HEADDIM,
        BLOCK_M = BLOCK,
        BLOCK_N = BLOCK,
        num_warps = num_warps,
        num_stages = 1,
    )

    return o, m, lse

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
        assert not exists(mask) or padded_false_on_right_side(mask), 'key padding mask must only contain True (attend) on the left hand side, and False (not attend) on the right'

        dtype = q.dtype
        softmax_scale = q.shape[-1] ** -0.5

        if q.dtype == torch.float32:
            q = q.half()

        if k.dtype == torch.float32:
            k = k.half()

        if v.dtype == torch.float32:
            v = v.half()

        max_neg_value = -torch.finfo(dtype).max
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

        for (ring_rank, (is_first, is_last)), ((kv, mask), (receive_kv, receive_mask)) in ring_pass_fn(kv, mask, receive_buffers = (receive_kv, receive_mask), max_iters = max_ring_passes, ring_size = ring_size):
            k, v = kv

            # translate key padding mask to bias

            bias = None

            if exists(mask):
                bias = torch.where(mask, 0.,  max_neg_value)

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
                return_normalized_output = False,
                load_accumulated = not is_first
            )

        lse = lse[..., :q_seq_len]
        m = m[..., :q_seq_len]

        o_scale = torch.exp(m - lse)
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

        # hack for special causal mask for striped ring attention without having to modify cuda

        if causal and striped_ring_attn:
            # this is a hack that should also mask out the diagonal
            # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#21-change-behavior-of-causal-flag
            q = pad_at_dim(q, (0, 1), dim = 1)
            o = pad_at_dim(o, (0, 1), dim = 1)
            do = pad_at_dim(do, (0, 1), dim = 1)
            lse = pad_at_dim(lse, (0, 1), dim = -1)

        # if not causal and has key padding mask
        # prepare row related tensors with unpad_input

        if not causal and exists(mask):
            lse = rearrange(lse, 'b h n ... -> b n h ...')

            (
                (q, o, do, lse),
                cu_seqlens_q,
                cu_maxlen_q,
                repad_q
            ) = unpad_inputs_and_return_inverse_fn(
                (q, o, do, lse),
                mask
            )

        for (ring_rank, _), ((kv_and_dkv, mask), (receive_kv_and_dkv, receive_mask)) in ring_pass_fn(kv_and_dkv, mask, receive_buffers = (receive_kv_and_dkv, receive_mask), max_iters = max_ring_passes, ring_size = ring_size):

            kv, dk, dv = kv_and_dkv

            # reconstitute correct types for k, v, dk, dv

            k, v = kv.chunk(2, dim = -1)
            k, v = k.view(k_dtype), v.view(v_dtype)

            # determine whether to do causal mask or not
            # depends on whether it is striped attention, as well as current machine rank vs ring rank

            n = row_length

            if causal or not exists(mask):

                block_causal = False
                need_accum = True

                if causal:
                    if striped_ring_attn:
                        block_causal = True

                        if get_rank() < ring_rank:
                            n += 1
                    else:
                        block_causal = get_rank() == ring_rank

                        if get_rank() < ring_rank:
                            need_accum = False

                # use flash attention backwards kernel to calculate dq, dk, dv and accumulate

                if need_accum:
                    ring_dq, ring_dk, ring_dv, *_ = _flash_attn_backward(
                        dout = do[:, :n],
                        q = q[:, :n],
                        k = k,
                        v = v,
                        out = o[:, :n],
                        softmax_lse = lse[..., :n],
                        dq = torch.empty_like(q[:, :n]),
                        dk = torch.empty_like(k),
                        dv = torch.empty_like(v),
                        dropout_p = 0.,
                        softmax_scale = softmax_scale,
                        causal = block_causal,
                        window_size = (-1, -1),
                        alibi_slopes = None,
                        deterministic = False
                    )

                    ring_dq = ring_dq[:, :row_length]

                else:
                    ring_dq, ring_dk, ring_dv = 0., 0., 0.

                q = q[:, :row_length]
                o = o[:, :row_length]
                do = do[:, :row_length]
                lse = lse[..., :row_length]

            else:

                (
                    (k, v),
                    cu_seqlens_k,
                    cu_maxlen_k,
                    repad_kv
                ) = unpad_inputs_and_return_inverse_fn(
                    (k, v),
                    mask
                )

                if not is_empty(q) and not is_empty(k):
                    ring_dq, ring_dk, ring_dv, *_ = _flash_attn_varlen_backward(
                        dout = do,
                        q = q,
                        k = k,
                        v = v,
                        out = o,
                        softmax_lse = lse,
                        dq = torch.empty_like(q),
                        dk = torch.empty_like(k),
                        dv = torch.empty_like(v),
                        cu_seqlens_q = cu_seqlens_q,
                        cu_seqlens_k = cu_seqlens_k,
                        max_seqlen_q = cu_maxlen_q,
                        max_seqlen_k = cu_maxlen_k,
                        dropout_p = 0.,
                        softmax_scale = softmax_scale,
                        causal = False,
                        window_size = (-1, -1),
                        alibi_slopes = None,
                        deterministic = False
                    )

                    ring_dq = repad_q(ring_dq)
                    ring_dk = repad_kv(ring_dk)
                    ring_dv = repad_kv(ring_dv)

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
