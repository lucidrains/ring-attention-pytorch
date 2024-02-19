from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einx import rearrange

from ring_attention_pytorch.ring import (
    all_ring_pass,
    is_distributed,
    get_rank,
    get_world_size
)

from ring_attention_pytorch.ring_flash_attention import (
    ring_flash_attn
)

from ring_attention_pytorch.distributed import (
    split_by_rank,
    AllGather
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def default_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor],
    causal: bool = False
):
    q = q * (q.shape[-1] ** 0.5)

    mask_value = -torch.finfo(q.dtype).max

    # similarity

    sim = einx.dot('b h i d, b h j d -> b h i j', q, k)

    # masking

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
        sim = einx.where('i j, , b h i j -> b h i j', causal_mask, mask_value, sim)

    elif exists(mask):
        sim = einx.where('b j, b h i j, -> b h i j', mask, sim, mask_value)

    # attend

    attn = einx.softmax('b h i [j]', sim)

    # aggregate

    out = einx.dot('b h i j, b h j d -> b h i d', attn, v)

    return out

# batch to sequence sharding and back

def pad_to_multiple(
    x: Tensor,
    length: int,
    pad_value = 0
):
    seq_len = x.shape[-1]
    remainder = seq_len % length

    if remainder == 0:
        return x, 0

    pad_length = length - remainder
    return F.pad(x, (0, pad_length), value = pad_value), pad_length

def maybe_pad_seq_and_mask(
    x: Tensor,
    mask: Optional[Tensor],
    seq_size: int
):
    orig_x, seq_len = x, x.shape[-1]

    # auto pad sequence and mask, as ring passing makes assumption tensor is all same shape

    x, pad_length = pad_to_multiple(x, seq_size)

    if pad_length == 0:
        return x, mask

    if not exists(mask):
        mask = torch.ones_like(orig_x).bool()

    mask, _ = pad_to_multiple(mask, seq_size, pad_value = False)

    return x, mask

def sharded_batch_to_sharded_seq(
    x: Tensor,
    mask: Optional[Tensor],
    seq_size: int
):
    assert is_distributed()

    # all gather across batch

    all_gather = AllGather(dim = 0)

    x, sizes = all_gather(x)

    if exists(mask):
        mask, _ = all_gather(mask)

    # then split sequence across machines

    x = x.split(seq_size, dim = -1)

    assert len(x) == get_world_size()

    x, _ = split_by_rank(x)

    if exists(mask):
        mask = mask.split(seq_size, dim = -1)
        mask, _ = split_by_rank(mask)

    return (x, mask), sizes

def sharded_seq_to_sharded_batch(
    logits: Tensor,
    sizes
):
    all_gather = AllGather(dim = -2) # all gather across sequence

    logits, _ = all_gather(logits)

    logits = logits.split(sizes.tolist(), dim = 0)

    logits = split_by_rank(logits)

    return logits

# main class

class RingAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False,
        eps = 1e-10,
        bucket_size = 512,
        ring_attn = False,
        ring_seq_size = 512,
        striped_ring_attn = False,
        auto_shard_seq = None,
        prenorm = True,
        force_regular_attn = False,
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        assert divisible_by(ring_seq_size, bucket_size)

        self.ring_attn = ring_attn
        self.striped_ring_attn = striped_ring_attn

        self.force_regular_attn = force_regular_attn
        self.auto_shard_seq = default(auto_shard_seq, ring_attn) # this should be done at the transformer level on the token ids for efficiency, but for testing purposes

        assert not (not self.ring_attn and self.auto_shard_seq)

        self.ring_seq_size = ring_seq_size

        self.bucket_size = bucket_size

        dim_inner = dim_head * heads
        self.to_qkv = nn.Sequential(
            RMSNorm(dim) if prenorm else nn.Identity(),
            nn.Linear(dim, dim_inner * 3, bias = False)
        )

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

        ring_attn = self.ring_attn & is_distributed()
        auto_shard_seq = self.auto_shard_seq & is_distributed()

        seq_len = x.shape[-1]

        if auto_shard_seq:
            (x, mask), batch_sizes = sharded_batch_to_sharded_seq(x, mask, self.ring_seq_size)

        device = x.device

        qkv = self.to_qkv(x)
        q, k, v = rearrange('b n (qkv h d) -> qkv b h n d', qkv, qkv = 3, h = self.heads)

        if self.force_regular_attn or not is_distributed():
            out = default_attention(q, k, v, mask = mask, causal = self.causal)
        else:
            out = ring_flash_attn(
                q, k, v,
                mask,
                self.causal,
                self.bucket_size,
                ring_attn,
                self.striped_ring_attn
            )

        # combine heads

        out = rearrange('b h n d -> b n (h d)', out)
        out = self.to_out(out)

        if auto_shard_seq:
            out, _ = sharded_seq_to_sharded_batch(out, batch_sizes)
            out = out[:, :seq_len]

        return out

# simple transformer for end2end testing

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

class RingTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        bucket_size = 512,
        ring_attn = False,
        striped_ring_attn = False,
        ring_seq_size = 512,
        auto_shard_seq = None,
    ):
        super().__init__()
        self.ring_attn = ring_attn
        self.striped_ring_attn = striped_ring_attn

        self.ring_seq_size = ring_seq_size
        self.auto_shard_seq = default(auto_shard_seq, ring_attn) # if ring attention is turned on, auto-shard across sequence dimension. this can also be turned off and done manually elsewhere in the data loading

        assert not (not self.ring_attn and self.auto_shard_seq)
        assert not (not self.ring_attn and self.striped_ring_attn)

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                RingAttention(
                    dim = dim,
                    causal = causal,
                    dim_head = dim_head,
                    heads = heads,
                    bucket_size = bucket_size,
                    ring_attn = ring_attn,
                    ring_seq_size = ring_seq_size,
                    striped_ring_attn = striped_ring_attn,
                    auto_shard_seq = False,
                ),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(
        self,
        x,
        mask = None
    ):
        seq_len = x.shape[-1]
        auto_shard_seq = self.auto_shard_seq & is_distributed()

        # take care of padding to divide sequence across the machines

        if auto_shard_seq:

            # first pad to right multiple

            x, mask = maybe_pad_seq_and_mask(x, mask, self.ring_seq_size)

            # account for striped attention
            # for workload balancing https://arxiv.org/abs/2311.09431 - MIT paper from Brandon et al.

            if self.striped_ring_attn:
                x = rearrange('b (i j) -> b (j i)', x, i = self.ring_seq_size)

                if exists(mask):
                    mask = rearrange('b (i j) -> b (j i)', mask, i = self.ring_seq_size)

            # gather across batch and divide across world

            (x, mask), batch_sizes = sharded_batch_to_sharded_seq(x, mask, self.ring_seq_size)

        # main transformer logic

        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        logits = self.to_logits(x)

        # now gather all sequence chunks across machines and shard back to original batch for cross entropy loss

        if auto_shard_seq:

            logits, _ = sharded_seq_to_sharded_batch(logits, batch_sizes)

            if self.striped_ring_attn:
                logits = rearrange('b (i j) d -> b (j i) d', logits, j = self.ring_seq_size)

            logits = logits[:, :seq_len]

        return logits
