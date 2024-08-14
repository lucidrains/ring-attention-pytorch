import torch
from torch import einsum
import torch.distributed as dist

from ring_attention_pytorch import tree_attn_decode

# regular attention for testing

def regular_decode(q, k, v):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    sim = einsum('... i d, ... j d -> ... i j', q, k)
    attn = sim.softmax(dim = -1)
    return einsum('... i j, ... j d -> ... i d', attn, v)

# for testing the above tree decoding function
# `pip install click` as requirement, besides `torch`

import os
import click
from math import ceil

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(
    rank,
    world_size,
    use_cuda
):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    backend = "gloo" if not use_cuda else "nccl"
    dist.init_process_group(backend, rank = rank, world_size = world_size)

    if use_cuda:
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def start(
    rank,
    world_size,
    dim,
    heads,
    batch,
    seq_len,
    use_cuda,
):
    setup(rank, world_size, use_cuda)
    is_main = rank == 0

    ring_seq_size = ceil(seq_len / world_size)

    # inputs

    q = torch.randn(batch, heads, 1, dim).half()
    k = torch.randn(batch, heads, seq_len, dim).half()
    v = torch.randn(batch, heads, seq_len, dim).half()

    if use_cuda:
        q, k, v = tuple(t.cuda(rank) for t in (q, k, v))

    # easy forcing all q, k, v to be same across all device

    dist.all_reduce(q)
    dist.all_reduce(k)
    dist.all_reduce(v)

    # outputs

    out = regular_decode(q, k, v)
    tree_out = tree_attn_decode(q, k, v)

    out = out.to(tree_out.dtype)

    # if not main early return

    if not is_main:
        return cleanup()

    # if is main, validate output is the same for kv sequence split across machines vs without

    tree_out = tree_out.cpu()
    out = out.cpu()

    output_atol = 1e-2 if use_cuda else 1e-5

    assert torch.allclose(tree_out, out, atol = output_atol), '🟥 output is not the same'

    print('✅ output is the same between tree and non-tree attention decoding')

    cleanup()

@click.command()
@click.option('--world-size', default = 8, help = 'number of machines / processes')
@click.option('--dim', default = 64, help = 'dimension')
@click.option('--heads', default = 8, help = 'dimension')
@click.option('--batch', default = 1, help = 'dimension')
@click.option('--use-cuda', is_flag = True, help = 'whether to test with CUDA and NCCL')
@click.option('--seq-len', default = 31, help = 'sequence length to test')
def test(
    world_size: int,
    dim: int,
    heads: int,
    batch: int,
    use_cuda: bool,
    seq_len: int,
):
    assert not use_cuda or world_size <= torch.cuda.device_count(), f'world size {world_size} must be less than the number of cuda devices {torch.cuda.device_count()}'

    mp.spawn(
        start,
        args = (world_size, dim, heads, batch, seq_len, use_cuda),
        nprocs = world_size,
        join = True
    )

if __name__ == '__main__':
    test()
