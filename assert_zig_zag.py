import os
import click
from math import ceil

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from ring_attention_pytorch import RingAttention
from ring_attention_pytorch.distributed import all_gather_variable_dim

from einops import rearrange

from ring_attention_pytorch.ring_attention import apply_rotary_pos_emb

from ring_attention_pytorch.zig_zag_attention import (
    zig_zag_pad_seq,
    zig_zag_attn,
    zig_zag_shard
)

def abs_diff(x, y):
    return (x - y).abs().amax()

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
    num_sharded_batches,
    dim,
    heads,
    num_grouped_query_heads,
    dim_head,
    use_cuda,
    rotary
):
    setup(rank, world_size, use_cuda)

    attention = RingAttention(
        dim = dim,
        dim_head = dim_head,
        heads = heads,
        num_grouped_query_heads = num_grouped_query_heads,
        causal = True,
        rotary_embed = rotary,
        ring_attn = False,
        use_cuda_kernel = use_cuda
    )

    if batch_size_var_len:
        batch_size = batch_size + rank

    seq = torch.randn(batch_size, seq_len, dim)

    # move to cuda if needed

    if use_cuda:
        seq = seq.cuda(rank)
        attention.cuda(rank)

    # separate inputs for ring vs flash

    regular_input = seq.clone().requires_grad_()
    zig_zag_input = seq.clone().requires_grad_()

    # wrap

    ddp_attention = DDP(attention)

    # regular

    out = ddp_attention(regular_input)

    out.mean().backward()

    # zig zag

    padded_inp, remove_pad = zig_zag_pad_seq(zig_zag_input)
    (padded_inp, q_indices, kv_indices), gather_seq = zig_zag_shard(padded_inp, all_gather_batch = True)

    qkv = attention.to_qkv(padded_inp)

    q, k, v = rearrange(qkv, 'b n (h d) -> b h n d', d = dim_head).split(attention.qkv_head_breakdown, dim = -3)

    if rotary:
        pos_emb = attention.rotary_embed(q_indices)

        q = apply_rotary_pos_emb(pos_emb, q, head_dim_first = True)
        k = apply_rotary_pos_emb(pos_emb, k, head_dim_first = True)

    # causal mask

    causal_mask = q_indices[:, None] >= kv_indices[None, :]

    # attention

    o = zig_zag_attn(
        q, k, v,
        attn_mask = causal_mask
    )

    o = rearrange(o, 'b h n d -> b n (h d)')

    padded_out = attention.to_out(o)

    padded_out = gather_seq(padded_out)

    zig_zag_out = remove_pad(padded_out)

    zig_zag_out.mean().backward()

    # validate output is the same for sequence split across machines vs without

    if rank == 0:
        out = out.cpu()
        zig_zag_out = zig_zag_out.cpu()

        output_atol = 1e-2 if use_cuda else 1e-6

        assert torch.allclose(out, zig_zag_out, atol = output_atol), 'output is not the same'

        # validate gradients is the same

        regular_input_grad = regular_input.grad
        zig_zag_input_grad = zig_zag_input.grad

        assert torch.allclose(
            regular_input_grad,
            zig_zag_input_grad,
            atol = 1e-2
        ), 'grad is not the same'

        print('✅ outputs and gradients are same between zig zag attention and regular attention')

    cleanup()

@click.command()
@click.option('--world-size', default = 8, help = 'number of machines / processes')
@click.option('--batch-size', default = 2, help = 'test batch size')
@click.option('--num-sharded-batches', default = 1, help = 'number of sharded batches')
@click.option('--batch-size-var-len', is_flag = True, help = 'test variable lengthed batch sizes')
@click.option('--use-cuda', is_flag = True, help = 'whether to test with CUDA and NCCL')
@click.option('--rotary', is_flag = True, help = 'whether to test with rotary embeddings')
@click.option('--seq-len', default = 31, help = 'sequence length to test')
@click.option('--model-dim', default = 8, help = 'model dimensions for testing')
@click.option('--heads', default = 8, help = 'number of query attention heads')
@click.option('--num-grouped-query-heads', default = 2, help = 'number of query attention head groups')
@click.option('--dim-head', default = 16, help = 'model dimensions for testing')
def test(
    world_size: int,
    batch_size: int,
    num_sharded_batches: int,
    batch_size_var_len: bool,
    use_cuda: bool,
    rotary: bool,
    seq_len: int,
    model_dim: int,
    heads: int,
    num_grouped_query_heads: int,
    dim_head: int,
):
    assert not use_cuda or world_size <= torch.cuda.device_count(), f'world size {world_size} must be less than the number of cuda devices {torch.cuda.device_count()}'

    mp.spawn(
        start,
        args = (
            world_size,
            batch_size,
            batch_size_var_len,
            seq_len,
            num_sharded_batches,
            model_dim,
            heads,
            num_grouped_query_heads,
            dim_head,
            use_cuda,
            rotary
        ),
        nprocs = world_size,
        join = True
    )

if __name__ == '__main__':
    test()
