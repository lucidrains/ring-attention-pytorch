import torch
from torch import nn, einsum
from torch.nn import Module
from torch.cuda.amp import autocast

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        theta = 10000
    ):
        super().__init__()
        inv_freq = theta ** -(torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    @autocast(enabled = False)
    def forward(
        self,
        pos,
        offset = 0
    ):
        pos = pos.type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', pos, self.inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim=-1)

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()
