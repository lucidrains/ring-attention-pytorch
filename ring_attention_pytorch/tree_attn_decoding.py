from __future__ import annotations

import torch
from torch import einsum, Tensor
import torch.distributed as dist

from ring_attention_pytorch.distributed import get_rank, get_world_size

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main function

@torch.no_grad()
def tree_attn_decode(
    q: Tensor,
    k: Tensor | None = None,
    v: Tensor | None = None,
    eps = 1e-8,
    shard_kv_seq = False,
    use_triton = None
):
    dtype = q.dtype

    assert not (exists(k) ^ exists(v)), 'keys and values are either both None, or both present'

    if exists(k):
        assert k.shape[:-1] == v.shape[:-1]
        assert q.shape[-2:] == (1, k.shape[-1])
        assert q.shape[:-2] == k.shape[:-2]

    """
    Algorithm 3 proposed in Tree Attention
    https://arxiv.org/abs/2408.04093
    """

    # each machine (rank) takes care of a chunk of kv sequence within the world of many machines

    if shard_kv_seq:
        assert exists(k), 'keys and values must be passed if not already sharded across sequence'

        rank, world_size = get_rank(), get_world_size()
        k = k.chunk(world_size, dim = -2)
        v = v.chunk(world_size, dim = -2)

        k, v = (k[rank], v[rank]) if rank < len(k) else (None, None)

    if exists(k):
        # calculate local output and derive numerator and denominator

        use_triton = default(use_triton, q.is_cuda)
        assert not (use_triton and not q.is_cuda), 'input needs to be on cuda if forcing the use of triton'

        if use_triton and q.is_cuda:
            from ring_attention_pytorch.triton_flash_attn import flash_attn_forward

            local_out, _, lse = flash_attn_forward(
                q, k, v,
                causal = False,
                return_normalized_output = True,
                load_accumulated = False,
                head_first_dim = True,
                remove_padding = True
            )

        else:
            scale = q.shape[-1] ** -0.5
            sim = einsum('... i d, ... j d -> ... i j', q, k) * scale

            lse = sim.logsumexp(dim = -1, keepdim = True)
            attn = sim.softmax(dim = -1)
            local_out = einsum('... i j, ... j d -> ... i d', attn, v)

    else:
        # handle edge case where seq length < world size

        local_out = q.new_zeros((*q.shape[:-1], v.shape[-1]), dtype = torch.float32)
        lse = torch.full_like(den, -torch.finfo(torch.float32).max)

    # first get global max through an all reduce (max)

    global_lse = lse.clone()
    dist.all_reduce(global_lse, dist.ReduceOp.MAX)

    # renormalize the numerator and denominators

    den = (lse - global_lse).exp()
    num = local_out * den

    # second and third all reduce (sum)

    dist.all_reduce(den)
    dist.all_reduce(num)

    out = num.div_(den.clamp(min = eps))

    return out.type(dtype)