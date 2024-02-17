from functools import lru_cache, partial, wraps

import torch
from torch import nn
from torch.nn import Module, ModuleList
from torch.autograd import Function
import torch.distributed as dist

import einx
from einx import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def maybe(fn):
    @wraps(fn)
    def inner(t, *args, **kwargs):
        if not exists(t):
            return None
        return fn(t, *args, **kwargs)
    return inner

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

def circular_index_left(pos, ring_size):
    return ((pos - 1) + ring_size) % ring_size

def circular_index_right(pos, ring_size):
    return (pos + 1) % ring_size

# distributed ring

def circular_rank_left(rank = None, ring_size = None):
    rank = default(rank, get_rank())
    ring_size = default(ring_size, get_world_size())
    return circular_index_left(rank, ring_size)

def circular_rank_right(rank = None, ring_size = None):
    rank = default(rank, get_rank())
    ring_size = default(ring_size, get_world_size())
    return circular_index_right(rank, ring_size)

# one ring pass

def send_and_receive_(x, receive_buffer, send_to_rank, receive_from_rank):
    send_request = dist.isend(x, send_to_rank)
    dist.recv(receive_buffer, receive_from_rank)

    send_request.wait()
    dist.barrier()

class OneRingPass(Function):
    """ one ring pass to the right - assume tensor is all same shape for now """

    @staticmethod
    def forward(ctx, x):
        receive_buffer = torch.zeros_like(x)
        send_and_receive_(x, receive_buffer, circular_rank_right(), circular_rank_left())
        return receive_buffer

    @staticmethod
    def backward(ctx, grads):
        receive_buffer = torch.zeros_like(grads)
        send_and_receive_(grads, receive_buffer, circular_rank_left(), circular_rank_right())
        return receive_buffer

one_ring_pass = OneRingPass.apply

# iterator for all ring passes of all tensors

def null_ring_pass(*tensors):
    yield 0, tuple(tensors)

def all_ring_pass(*tensors):
    num_passes = 0
    curr_rank = get_rank()
    world_size = get_world_size()

    while num_passes < world_size:

        curr_rank = (curr_rank + num_passes) % world_size

        yield curr_rank, tuple(tensors)

        num_passes += 1

        if num_passes >= world_size:
            continue

        tensors = tuple(map(maybe(one_ring_pass), tensors))