import os
from math import ceil
from copy import deepcopy

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ring_attention_pytorch.ring_attention import RingTransformer
from ring_attention_pytorch.distributed import all_gather_variable_dim

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank = rank, world_size = world_size)

def cleanup():
    dist.destroy_process_group()

def start(
    rank,
    world_size,
    batch_size,
    batch_size_var_len,
    seq_len,
    causal,
    dim,
    use_cuda
):
    setup(rank, world_size)

    ring_attention_net = RingTransformer(
        num_tokens = 256,
        dim = dim,
        causal = causal,
        depth = 1,
        dim_head = 8,
        ring_attn = True,
        ring_seq_size = ceil(seq_len / world_size),
        q_bucket_size = ceil(seq_len / world_size),
        k_bucket_size = ceil(seq_len / world_size)
    )

    flash_attention_net = RingTransformer(
        num_tokens = 256,
        dim = dim,
        causal = causal,
        depth = 1,
        dim_head = 8,
        ring_attn = False
    )

    flash_attention_net.load_state_dict(ring_attention_net.state_dict())

    if batch_size_var_len:
        batch_size = batch_size + rank

    seq = torch.randint(0, 256, (batch_size, seq_len))

    # wrap

    ddp_ring_attention_net = DDP(ring_attention_net)
    ddp_flash_attention_net = DDP(flash_attention_net)

    inputs, _ = all_gather_variable_dim(seq)

    if use_cuda:
        inputs = inputs.cuda(rank)
        flash_attention_net.cuda(rank)
        ring_attention_net.cuda(rank)

    # flash

    flash_out = ddp_flash_attention_net(inputs)

    flash_out.mean().backward()

    # ring

    ring_out = ddp_ring_attention_net(inputs)

    ring_out.mean().backward()

    # validate output is the same for local vs distributed

    if rank == 0:

        ring_attention_net = ring_attention_net.cpu()
        flash_attention_net = flash_attention_net.cpu()
        ring_out = ring_out.cpu()
        flash_out = flash_out.cpu()

        assert torch.allclose(ring_out, flash_out, atol = 1e-6), 'output is not the same'

        # validate gradients of token embedding is the same for ring vs non-ring

        get_embed_grad = lambda model: model.token_emb.weight.grad
        ring_embed_grad = get_embed_grad(ring_attention_net)
        flash_embed_grad = get_embed_grad(flash_attention_net)

        assert torch.allclose(
            ring_embed_grad,
            flash_embed_grad,
            atol = 1e-3
        ), 'grad is not the same'

        print('✅ outputs and gradients are same between ring attention and non-ring attention')

    cleanup()

if __name__ == '__main__':
    world_size = 8
    batch_size = 1
    batch_size_var_len = False
    use_cuda = False
    causal = True

    assert not use_cuda or torch.cuda.device_count() <= world_size

    seq_len = 31
    dim = 8

    mp.spawn(
        start,
        args = (
            world_size,
            batch_size,
            batch_size_var_len,
            seq_len,
            causal,
            dim,
            use_cuda
        ),
        nprocs = world_size,
        join = True
    )
