import torch
from torch import einsum
import torch.distributed as dist

def tree_attn_decode(q, k, v):
    """
    Algorithm 3 proposed in Tree Attention
    https://arxiv.org/abs/2408.04093
    """

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # scale queries

    scale = q.shape[-1] ** -0.5
    q = q * scale

    # each machine (rank) takes care of a chunk of kv sequence within the world of many machines

    k = k.chunk(world_size, dim = -2)
    v = v.chunk(world_size, dim = -2)

    k, v = k[rank], v[rank]

    # first calculate local output

    sim = einsum('... i d, ... j d -> ... i j', q, k)

    local_max = sim.amax(dim = -1, keepdim = True)
    sim = sim - local_max
    lse = sim.logsumexp(dim = -1, keepdim = True)

    attn = sim.softmax(dim = -1)
    out = einsum('... i j, ... j d -> ... i d', attn, v)

    den = lse.exp()
    num = out * den

    # first get global max through an all reduce (max)

    global_max = local_max.clone()
    dist.all_reduce(global_max, dist.ReduceOp.MAX)

    # renormalize the numerator and denominators

    renorm_factor = (local_max - global_max).exp()

    den = den * renorm_factor
    num = num * renorm_factor

    # second and third all reduce (sum)

    dist.all_reduce(den)
    dist.all_reduce(num)

    return num / den
