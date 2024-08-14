import torch
from torch import einsum
import torch.distributed as dist

from ring_attention_pytorch.distributed import get_rank, get_world_size

def exists(v):
    return v is not None

@torch.no_grad()
def tree_attn_decode(
    q, k, v,
    eps = 1e-8,
    shard_kv_seq = False
):

    assert k.shape[:-1] == v.shape[:-1]
    assert q.shape[-2:] == (1, k.shape[-1])
    assert q.shape[:-2] == k.shape[:-2]

    """
    Algorithm 3 proposed in Tree Attention
    https://arxiv.org/abs/2408.04093
    """

    device, dim_v = q.device, v.shape[-1]

    rank = get_rank()
    world_size = get_world_size()

    # scale queries

    scale = q.shape[-1] ** -0.5
    q *= scale

    # each machine (rank) takes care of a chunk of kv sequence within the world of many machines

    if shard_kv_seq:
        k = k.chunk(world_size, dim = -2)
        v = v.chunk(world_size, dim = -2)
        k, v = (k[rank], v[rank]) if rank < len(k) else (None, None)

    if exists(k) and exists(v):
        # calculate local output and derive numerator and denominator

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        local_max = sim.amax(dim = -1, keepdim = True)
        sim -= local_max
        lse = sim.logsumexp(dim = -1, keepdim = True)

        attn = sim.softmax(dim = -1)
        out = einsum('... i j, ... j d -> ... i d', attn, v)

        den = lse.exp()
        num = out * den

    else:
        # handle edge case where seq length < world size

        num = q.new_zeros((*q.shape[:-1], dim_v))
        den = q.new_zeros((*q.shape[:-1], 1))
        local_max = torch.zeros_like(den)

    # first get global max through an all reduce (max)

    global_max = local_max.clone()
    dist.all_reduce(global_max, dist.ReduceOp.MAX)

    # renormalize the numerator and denominators

    renorm_factor = (local_max - global_max).exp()

    den *= renorm_factor
    num *= renorm_factor

    # second and third all reduce (sum)

    dist.all_reduce(den)
    dist.all_reduce(num)

    return num / den.clamp(min = eps)
