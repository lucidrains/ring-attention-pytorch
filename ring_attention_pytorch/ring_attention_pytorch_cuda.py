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

if not exists(importlib.util.find_spec('flash-attn')):
    print('flash-attn must be installed. `pip install flash-attn --no-build-isolation` first')
    exit()

from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward
)

# make sure triton is installed for backwards

if not exists(importlib.util.find_spec('triton')):
    print('triton must be installed. `pip install triton` first')
    exit()

import triton
import triton.language as tl

# taking the flash attention forwards
# from https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py
# and modifying to return unnormalized accumulation, row maxes, row sums - reduced over passed rings

@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr
):

    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(tl.float16), v)

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    M,
    L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    q = tl.load(Q_block_ptr)

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX  #
                                        )
    if STAGE & 2:

        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX  #
                                        )
    # epilogue

    m_ptrs = M + off_hz * N_CTX + offs_m
    l_ptrs = L + off_hz * N_CTX + offs_l

    tl.store(m_ptrs, m_i)
    tl.store(l_ptrs, l_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

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

            # notation that researchers use is
            # o - output
            # m - row maxes
            # l - row sums
            # m and l is summarized into a single logsumexp (lse) at the end for backward

            ring_o = torch.empty_like(o)
            ring_m = torch.empty_like(all_row_maxes)
            ring_l = torch.empty_like(all_row_sums)

            _attn_fwd(
                q, k, v
                scale,
                ring_l,
                ring_m,
                ring_o,
            )

            o.add_(ring_o)
            all_row_sums.add_(all_row_sums)
            all_row_maxes = torch.maximum(all_row_maxes, ring_m)

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
