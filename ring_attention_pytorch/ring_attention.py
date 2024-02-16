from functools import lru_cache, partial

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

def maybe_sum(*args):
    args = [*filter(exists, args)]
    if len(args) == 0:
        return None
    return sum(args)

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

# main class

class RingAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False,
        eps = 1e-10
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        dim_inner = dim_head * heads
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        mask = None
    ):
        """
        einstein notation

        b - batch
        h - heads
        d - feature dimension
        n, i, j - sequence
        """
        device, mask_value = x.device, -torch.finfo(x.dtype).max

        qkv = self.to_qkv(x)
        q, k, v = rearrange('b n (qkv h d) -> qkv b h n d', qkv, qkv = 3, h = self.heads)

        q = q * self.scale

        if not is_distributed():
            # similarity

            sim = einx.dot('b h i d, b h j d -> b h i j', q, k)

            # masking

            if self.causal:
                i, j = sim.shape[-2:]
                causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
                sim = einx.where('i j, , b h i j -> b h i j', causal_mask, mask_value, sim)

            elif exists(mask):
                sim = einx.where('b j, b h i j, -> b h i j', mask, sim, mask_value)

            # attend

            attn = einx.softmax('b h i [j]', sim)

            # aggregate

            out = einx.dot('b h i j, b h j d -> b h i d', attn, v)

        else:
            # accumulate outputs numerator and denominator

            num_passes = 0

            out = torch.zeros_like(q)
            row_sums = torch.zeros((*q.shape[:-1], 1), device = device)
            row_maxes = torch.full((*q.shape[:-1], 1), mask_value, device = device)

            while num_passes < get_world_size():

                attn_weights = einx.dot('... i d, ... j d -> ... i j', q, k)

                if exists(mask):
                    attn_weights = einx.where('b j, b h i j, -> b h i j', mask, attn_weights, mask_value)

                block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_weights = torch.exp(attn_weights - new_row_maxes)

                if exists(mask):
                    exp_weights = einx.where('b j, b h i j,', mask, exp_weights, 0.)

                block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = self.eps)

                exp_values = einx.dot('... i j, ... j d -> ... i d', exp_weights, v)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)

                # update row sums and row maxes

                row_sums = exp_row_max_diff * row_sums + block_row_sums
                row_maxes = new_row_maxes

                # accumulate out

                out = out * exp_row_max_diff + exp_values

                # increment number of passes

                num_passes += 1

                if num_passes >= get_world_size():
                    continue

                k = one_ring_pass(k)
                v = one_ring_pass(v)

                if exists(mask):
                    mask = one_ring_pass(mask)

            out = out / row_sums

        # combine heads

        out = rearrange('b h n d -> b n (h d)', out)
        return self.to_out(out)
