from __future__ import annotations

from functools import lru_cache, partial, wraps
from collections import namedtuple

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch.autograd import Function

import torch.distributed as dist

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

cache = partial(lru_cache, maxsize = None)

# distributed globals

@cache()
def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

@cache()
def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

@cache()
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

# ring functions

def circular_index_left(pos, ring_size, num = 1):
    return ((pos - num) + ring_size) % ring_size

def circular_index_right(pos, ring_size, num = 1):
    return (pos + num) % ring_size

# distributed ring

def circular_rank_left(rank = None, ring_size = None, num = 1):
    rank = default(rank, get_rank())
    ring_size = default(ring_size, get_world_size())
    ring_set_num = rank // ring_size
    offset = ring_set_num * ring_size
    return circular_index_left(rank, ring_size, num) + offset

def circular_rank_right(rank = None, ring_size = None, num = 1):
    rank = default(rank, get_rank())
    ring_size = default(ring_size, get_world_size())
    ring_set_num = rank // ring_size
    offset = ring_set_num * ring_size
    return circular_index_right(rank, ring_size, num) + offset

# one ring pass

def send_and_receive_(x, receive_buffer, send_to_rank, receive_from_rank):
    send_op = dist.P2POp(dist.isend, x, send_to_rank)
    recv_op = dist.P2POp(dist.irecv, receive_buffer, receive_from_rank)

    reqs = dist.batch_isend_irecv([send_op, recv_op])

    for req in reqs:
        req.wait()

    dist.barrier()

def ring_pass(
    num_ring_passes: int,
    x: Tensor,
    receive_buffer: Tensor | None = None,
    ring_size: int | None = None
):
    ring_size = default(ring_size, get_world_size())
    x = x.contiguous()

    if not exists(receive_buffer):
        receive_buffer = torch.zeros_like(x)
    else:
        receive_buffer = receive_buffer.contiguous()

    send_and_receive_(x, receive_buffer, circular_rank_right(ring_size = ring_size), circular_rank_left(ring_size = ring_size))
    return receive_buffer, x

one_ring_pass = partial(ring_pass, 1)

# iterator for all ring passes of all tensors

RingInfo = namedtuple('RingInfo', ['ring_rank', 'iter_info'])

def null_ring_pass(*tensors, max_iters = None, receive_buffers = None, ring_size = None):
    yield RingInfo(0, (True, True)), (tensors, receive_buffers)

def all_ring_pass(*tensors, max_iters = None, receive_buffers = None, ring_size = None):
    ring_size = default(ring_size, get_world_size())
    max_iters = default(max_iters, ring_size)

    receive_buffers = cast_tuple(receive_buffers, len(tensors))

    # make sure iteration is between 1 and world size

    total_iters = max(1, min(ring_size, max_iters))

    curr_ring_pos = get_rank()

    for ind in range(total_iters):
        is_first = ind == 0
        is_last = ind == (total_iters - 1)

        yield RingInfo(curr_ring_pos, (is_first,  is_last)), (tensors, receive_buffers)

        curr_ring_pos = circular_index_left(curr_ring_pos, ring_size)

        if is_last:
            continue

        new_tensors = []
        new_receive_buffers = []

        for tensor, receive_buffer in zip(tensors, receive_buffers):
            if exists(tensor):
                new_tensor, new_receive_buffer = one_ring_pass(tensor, receive_buffer, ring_size)
            else:
                new_tensor, new_receive_buffer = None, None

            new_tensors.append(new_tensor)
            new_receive_buffers.append(new_receive_buffer)

        tensors = new_tensors
        receive_buffers = new_receive_buffers
