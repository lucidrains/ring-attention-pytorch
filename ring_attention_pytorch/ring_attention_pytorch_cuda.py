import math
from typing import Optional

import torch
from torch import nn, einsum, Tensor
from torch.autograd.function import Function

from ring_attention_pytorch.ring import (
    ring_pass,
    all_ring_pass,
    null_ring_pass,
    one_ring_pass,
    get_rank,
    get_world_size
)

# make sure flash attention is installed for forwards

import importlib
from importlib.metadata import version

assert exists(importlib.util.find_spec('flash_attn')), 'flash-attn must be installed. `pip install flash-attn==2.5.1.post1 --no-build-isolation` first'
assert version('flash_attn') == '2.5.1.post1'

from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward
)

# make sure triton is installed for backwards

assert exists(importlib.util.find_spec('triton')), 'specific version of triton must be installed. `pip install triton==2.0.0.dev20221202` first'
assert version('triton') == '2.0.0.dev20221202'

import triton
import triton.language as tl

# taking the flash attention forwards from Tri's flash_attn repository
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
# and modifying to return unnormalized accumulation, row maxes, row lse - reduced over passed rings

import math

import torch
import triton
import triton.language as tl

from einops import rearrange

def exists(v):
    return v is not None

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
    TMP,
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
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
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

    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n

    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m

    # maximum

    m_ptrs = M + off_hb * seqlen_q_rounded + offs_m

    m_i = tl.load(m_ptrs)

    # load lse

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m

    lse_i = tl.load(lse_ptrs)

    # load accumualted output

    offs_d = tl.arange(0, BLOCK_HEADDIM)

    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )

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
        qk += tl.dot(q, k, trans_b=True)

        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
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

        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
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

    # write back l and m

    tl.store(lse_ptrs, lse_i)

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

def is_contiguous(x):
    return x.stride(-1) == 1

def flash_attn_forward(
    q,
    k,
    v,
    bias = None,
    causal = False,
    o = None,
    m = None,
    lse = None,
    softmax_scale = None
):

    q, k, v = [rearrange(t, 'b h n d -> b n h d') for t in (q, k, v)]
    q, k, v = [x if is_contiguous(x) else x.contiguous() for x in (q, k, v)]

    device = q.device
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape

    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)

    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128

    max_neg_value = -torch.finfo(torch.float32).max

    if not exists(lse):
        lse = torch.full((batch, nheads, seqlen_q_rounded), max_neg_value, device=device, dtype=torch.float32)

    if not exists(m):
        m = torch.full((batch, nheads, seqlen_q_rounded), max_neg_value, dtype=device)

    if not exists(o):
        o = torch.zeros_like(q)

    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

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
        tmp,
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
        bias_type,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M = BLOCK,
        BLOCK_N = BLOCK,
        num_warps = num_warps,
        num_stages = 1,
    )

    return o, lse, m, softmax_scale

# helper functions

EPSILON = 1e-10

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

        num_tiles = math.ceil(per_machine_seq_size / bucket_size)

        kv = torch.stack((k, v))

        # accumulated values

        # o - output
        # m - maximum
        # lse - logsumexp
        # softmax_scale - scale changes as it is being reduced in new flash attention, do not totally understand the need for this just yet

        o = None
        m = None
        lse = None
        softmax_scale = None

        # receive buffers, to be alternated with sent buffer

        receive_kv = None
        receive_mask = None

        for ring_rank, ((kv, mask), (receive_kv, receive_mask)) in ring_pass_fn(kv, mask, receive_buffers = (receive_kv, receive_mask), max_iters = max_ring_passes, ring_size = ring_size):

            k, v = kv

            o, lse, m, softmax_scale = flash_attn_forward(
                q, k, v,
                causal = causal,
                o = o,
                m = m,
                lse = lse,
                softmax_scale = softmax_scale
            )

        # scale final output, taking into account running maximum and lse

        o_scale = torch.exp(m - lse)
        o.mul_(o_scale[:, None])

        ctx.args = (
            causal,
            softmax_scale,
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
            softmax_scale,
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
                softmax_scale = softmax_scale,
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
