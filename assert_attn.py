import os
import click
from math import ceil

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ring_attention_pytorch.ring_attention import RingAttention
from ring_attention_pytorch.distributed import all_gather_variable_dim

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
    batch_size,
    batch_size_var_len,
    seq_len,
    num_buckets,
    num_sharded_batches,
    causal,
    striped_ring_attn,
    dim,
    use_cuda,
    compare_regular_attn
):
    setup(rank, world_size, use_cuda)

    ring_seq_size = ceil(seq_len / world_size) * num_sharded_batches
    bucket_size = ring_seq_size // num_buckets

    ring_attention = RingAttention(
        dim = dim,
        causal = causal,
        dim_head = 8,
        ring_attn = True,
        striped_ring_attn = striped_ring_attn,
        ring_seq_size = ring_seq_size,
        bucket_size = bucket_size,
        use_cuda_kernel = use_cuda,
        auto_shard_seq = True
    )

    flash_attention = RingAttention(
        dim = dim,
        causal = causal,
        dim_head = 8,
        ring_attn = False,
        ring_seq_size = ring_seq_size,
        bucket_size = bucket_size,
        force_regular_attn = compare_regular_attn,
        use_cuda_kernel = False
    )

    flash_attention.load_state_dict(ring_attention.state_dict())

    if batch_size_var_len:
        batch_size = batch_size + rank

    seq = torch.randn(batch_size, seq_len, dim)

    # move to cuda if needed

    if use_cuda:
        seq = seq.cuda(rank)
        flash_attention_net.cuda(rank)
        ring_attention_net.cuda(rank)

    # separate inputs for ring vs flash

    flash_input = seq.clone().requires_grad_()
    ring_input = seq.clone().requires_grad_()

    # wrap

    ddp_ring_attention = DDP(ring_attention)
    ddp_flash_attention = DDP(flash_attention)

    # flash

    flash_out = ddp_flash_attention(flash_input)

    flash_out.mean().backward()

    # ring

    ring_out = ddp_ring_attention(ring_input)

    ring_out.mean().backward()

    # validate output is the same for sequence split across machines vs without

    if rank == 0:

        ring_attention = ring_attention.cpu()
        flash_attention = flash_attention.cpu()
        ring_out = ring_out.cpu()
        flash_out = flash_out.cpu()

        output_atol = 1e-2 if use_cuda else 1e-6

        assert torch.allclose(ring_out, flash_out, atol = output_atol), 'output is not the same'

        # validate gradients of token embedding is the same for ring vs non-ring

        ring_input_grad = ring_input.grad
        flash_input_grad = flash_input.grad

        assert torch.allclose(
            ring_input_grad,
            flash_input_grad,
            atol = 1e-2
        ), 'grad is not the same'

        print('✅ outputs and gradients are same between ring attention and non-ring attention')

    cleanup()

@click.command()
@click.option('--world-size', default = 8, help = 'number of machines / processes')
@click.option('--batch-size', default = 2, help = 'test batch size')
@click.option('--num-sharded-batches', default = 1, help = 'number of sharded batches')
@click.option('--batch-size-var-len', is_flag = True, help = 'test variable lengthed batch sizes')
@click.option('--use-cuda', is_flag = True, help = 'whether to test with CUDA and NCCL')
@click.option('--causal', is_flag = True, help = 'test autoregressive')
@click.option('--striped-ring-attn', is_flag = True, help = 'test striped ring attention from MIT follow up paper')
@click.option('--num-buckets', default = 2, help = 'number of buckets per machine (each sharded sequence is further windowed for flash attention to achieve even greater context lengths)')
@click.option('--seq-len', default = 31, help = 'sequence length to test')
@click.option('--model-dim', default = 8, help = 'model dimensions for testing')
@click.option('--compare-regular-attn', is_flag = True, help = 'compare ring to regular attention')
def test(
    world_size: int,
    batch_size: int,
    num_sharded_batches: int,
    batch_size_var_len: bool,
    use_cuda: bool,
    causal: bool,
    striped_ring_attn: bool,
    num_buckets: int,
    seq_len: int,
    model_dim: int,
    compare_regular_attn: bool
):
    assert not use_cuda or world_size <= torch.cuda.device_count(), f'world size {world_size} must be less than the number of cuda devices {torch.cuda.device_count()}'

    mp.spawn(
        start,
        args = (
            world_size,
            batch_size,
            batch_size_var_len,
            seq_len,
            num_buckets,
            num_sharded_batches,
            causal,
            striped_ring_attn,
            model_dim,
            use_cuda,
            compare_regular_attn
        ),
        nprocs = world_size,
        join = True
    )

if __name__ == '__main__':
    test()
