from typing import Optional, Tuple, Union

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import Module, ModuleList

import einx
from einx import rearrange

from beartype import beartype

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

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0

def default_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    causal: bool = False
):
    q = q * (q.shape[-1] ** -0.5)

    mask_value = -torch.finfo(q.dtype).max

    # similarity

    sim = einsum('b i h d, b j h d -> b h i j', q, k)

    # masking

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
        sim = torch.where(causal_mask, mask_value, sim)

    elif exists(mask):
        sim = einx.where('b j, b h i j, -> b h i j', mask, sim, mask_value)

    # attend

    attn = einx.softmax('b h i [j]', sim)

    # aggregate

    out = einsum('b h i j, b j h d -> b i h d', attn, v)

    return out

# rotary embeddings with modifications to support striped attention

class RingRotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        ring: bool = False,
        striped: bool = False,
        buckets: int = 1,        # in striped attention with flash buckets > 1, one needs to specify the number of buckets per machine
        theta = 10000
    ):
        super().__init__()
        self.ring = ring
        self.striped = striped
        self.buckets = buckets

        inv_freq = theta ** -(torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @autocast(enabled = False)
    def forward(
        self,
        seq_len: int,
        offset = 0
    ):
        device = self.device
        pos = None

        if self.ring:
            if self.striped:
                buckets = self.buckets
                ring_stride = get_world_size() * buckets
                ring_offset = buckets

                pos = torch.arange(seq_len // buckets, device = device)
                pos = rearrange('n -> n b', pos, b = buckets)

                pos = pos * ring_stride
                pos += torch.arange(buckets, device = device) + (get_rank() * buckets)
                pos = rearrange('n b -> (b n)', pos)

            else:
                pos = torch.arange(seq_len, device = device)
                pos += seq_len * get_rank()
        else:
            pos = torch.arange(seq_len, device = device)

        pos = pos.type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', pos, self.inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim=-1)

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    pos = rearrange('n d -> n 1 d', pos)
    return t * pos.cos() + rotate_half(t) * pos.sin()

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

    # first make sure world size is divisible by the sequence size

    world_size = get_world_size()

    total_split_seq = x.shape[-1] // seq_size

    assert divisible_by(world_size, total_split_seq)

    num_sharded_batches = world_size // total_split_seq

    x = rearrange('(b s) n -> b (s n)', x, s = num_sharded_batches)

    # then split sequence across machines

    x = x.split(seq_size, dim = -1)

    x, _ = split_by_rank(x)

    if exists(mask):
        mask = rearrange('(b s) n -> b (s n)', mask, s = num_sharded_batches)
        mask = mask.split(seq_size, dim = -1)
        mask, _ = split_by_rank(mask)

    return (x, mask), sizes, num_sharded_batches

def sharded_seq_to_sharded_batch(
    logits: Tensor,
    sizes,
    num_sharded_batches = 1
):
    all_gather = AllGather(dim = -2) # all gather across sequence

    logits, _ = all_gather(logits)

    logits = rearrange('b (s n) c -> (b s) n c', logits, s = num_sharded_batches)

    logits = logits.split(sizes.tolist(), dim = 0)

    logits, _ = split_by_rank(logits)

    return logits

# main class

class RingAttention(Module):
    @beartype
    def __init__(
        self,
        dim: int,
        *,
        dim_head: int = 64,
        heads: int = 8,
        causal: bool = False,
        eps: float = 1e-10,
        bucket_size: int = 512,
        ring_attn: bool = False,
        ring_seq_size: int = 512,
        max_lookback_seq_len: Optional[int] = None,
        striped_ring_attn: bool = False,
        auto_shard_seq: Optional[bool] = None,
        prenorm: bool = True,
        force_regular_attn: bool = False,
        rotary_embed: bool = False,
        rotary_embed_theta: int = 10000,
        use_cuda_kernel: bool = None
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        assert divisible_by(ring_seq_size, bucket_size)

        self.ring_attn = ring_attn
        self.max_lookback_seq_len = max_lookback_seq_len
        self.striped_ring_attn = striped_ring_attn

        self.force_regular_attn = force_regular_attn
        self.auto_shard_seq = default(auto_shard_seq, ring_attn) # this should be done at the transformer level on the token ids for efficiency, but for testing purposes

        assert not (not self.ring_attn and self.auto_shard_seq)

        self.ring_seq_size = ring_seq_size
        self.bucket_size = bucket_size

        # rotary

        self.rotary_embed = None
        if rotary_embed:
            self.rotary_embed = RingRotaryEmbedding(
                dim = dim_head,
                ring = ring_attn,
                striped = striped_ring_attn,
                theta = rotary_embed_theta,
                buckets = ring_seq_size // bucket_size
            )

        # projections

        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            RMSNorm(dim) if prenorm else nn.Identity(),
            nn.Linear(dim, dim_inner * 3, bias = False)
        )

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

        # whether to use flash attention cuda kernel

        self.use_cuda_kernel = default(use_cuda_kernel, torch.cuda.is_available())
        assert not (use_cuda_kernel and not torch.cuda.is_available())

    def forward(
        self,
        x,
        mask = None,
        rotary_emb = None,
        force_ring_reduce_off = False,
        ring_size = None,
    ):
        """
        einstein notation

        b - batch
        h - heads
        d - feature dimension
        n, i, j - sequence
        """

        ring_size = default(ring_size, get_world_size())
        ring_attn = self.ring_attn & is_distributed()
        auto_shard_seq = self.auto_shard_seq & is_distributed()

        seq_len = x.shape[-1]

        if auto_shard_seq:
            x, mask = maybe_pad_seq_and_mask(x, mask, self.ring_seq_size)

            if self.striped_ring_attn:
                x = rearrange('b (i j) d -> b (j i) d', x, i = self.bucket_size)

                if exists(mask):
                    mask = rearrange('b (i j) -> b (j i)', mask, i = self.bucket_size)

            (x, mask), batch_sizes = sharded_batch_to_sharded_seq(x, mask, self.ring_seq_size)

        device = x.device

        qkv = self.to_qkv(x)
        q, k, v = rearrange('b n (qkv h d) -> qkv b n h d', qkv, qkv = 3, h = self.heads)

        # rotary relative positions

        if not exists(rotary_emb) and exists(self.rotary_embed):
            rotary_emb = self.rotary_embed(q.shape[-2])

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        # regular attention vs flash w/ or w/o kv ring reduce

        any_cuda_inputs = any([t.is_cuda for t in (q, k, v)])

        if self.force_regular_attn:
            out = default_attention(q, k, v, mask = mask, causal = self.causal)

        elif any_cuda_inputs and self.use_cuda_kernel:
            from ring_attention_pytorch.ring_flash_attention_cuda import ring_flash_attn_cuda

            out = ring_flash_attn_cuda(
                q, k, v,
                mask,
                self.causal,
                self.bucket_size,
                ring_attn and not force_ring_reduce_off,
                self.striped_ring_attn and not force_ring_reduce_off,
                self.max_lookback_seq_len,
                ring_size
            )

        else:
            out = ring_flash_attn(
                q, k, v,
                mask,
                self.causal,
                self.bucket_size,
                ring_attn and not force_ring_reduce_off,
                self.striped_ring_attn and not force_ring_reduce_off,
                self.max_lookback_seq_len,
                ring_size
            )

        # combine heads

        out = rearrange('b n h d -> b n (h d)', out)
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
    @beartype
    def __init__(
        self,
        *,
        num_tokens: int,
        dim: int,
        depth: int,
        causal: bool = False,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        bucket_size: int = 512,
        ring_attn: bool = False,
        striped_ring_attn: bool = False,
        ring_seq_size: int = 512,
        auto_shard_seq: Optional[bool] = None,
        max_lookback_seq_len: Optional[Union[Tuple[int, ...], int]] = None,
        rotary_embed_theta: int = 10000,    # will need to be changed for the million token context
        ignore_index: int = -1
    ):
        super().__init__()
        self.ring_attn = ring_attn
        self.striped_ring_attn = striped_ring_attn

        self.ring_seq_size = ring_seq_size
        self.bucket_size = bucket_size
        assert divisible_by(ring_seq_size, bucket_size)

        self.auto_shard_seq = default(auto_shard_seq, ring_attn) # if ring attention is turned on, auto-shard across sequence dimension. this can also be turned off and done manually elsewhere in the data loading

        assert not (not self.ring_attn and self.auto_shard_seq)
        assert not (not self.ring_attn and self.striped_ring_attn)
        assert not (self.striped_ring_attn and not causal), 'striped ring attention only applies to autoregressive models'

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.rotary_emb = RingRotaryEmbedding(
            dim = dim_head,
            ring = ring_attn,
            striped = striped_ring_attn,
            theta = rotary_embed_theta,
            buckets = ring_seq_size // bucket_size
        )

        self.layers = ModuleList([])

        max_lookback_seq_len = cast_tuple(max_lookback_seq_len, depth)
        assert len(max_lookback_seq_len) == depth

        for layer_max_lookback_seq_len in max_lookback_seq_len:

            self.layers.append(ModuleList([
                RingAttention(
                    dim = dim,
                    causal = causal,
                    dim_head = dim_head,
                    heads = heads,
                    bucket_size = bucket_size,
                    ring_attn = ring_attn,
                    ring_seq_size = ring_seq_size,
                    max_lookback_seq_len = layer_max_lookback_seq_len,
                    striped_ring_attn = striped_ring_attn,
                    auto_shard_seq = False,
                ),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        # training related

        self.ignore_index = ignore_index

    def forward(
        self,
        x,
        mask = None,
        labels = None,
        return_loss = False,
        force_ring_reduce_off = False,
        ring_size = None
    ):
        seq_len, device = x.shape[-1], x.device

        auto_shard_seq = not force_ring_reduce_off and self.auto_shard_seq and is_distributed()

        # get labels if not passed in

        return_loss |= exists(labels)

        if return_loss and not exists(labels):
            x, labels = x[:, :-1], x[:, 1:]

        # take care of padding to divide sequence across the machines

        ring_size = default(ring_size, get_world_size())

        if auto_shard_seq:
            # first pad to right multiple

            x, mask = maybe_pad_seq_and_mask(x, mask, self.ring_seq_size)

            # labels

            if exists(labels):
                labels, label_mask = maybe_pad_seq_and_mask(labels, mask[:, 1:], self.ring_seq_size)
                labels.masked_fill_(~label_mask, self.ignore_index)

            # account for striped attention
            # for workload balancing https://arxiv.org/abs/2311.09431 - MIT paper from Brandon et al.

            if self.striped_ring_attn:
                x = rearrange('b (i j) -> b (j i)', x, i = self.bucket_size)

                if exists(labels):
                    labels = rearrange('b (i j) -> b (j i)', labels, i = self.bucket_size)

                if exists(mask):
                    mask = rearrange('b (i j) -> b (j i)', mask, i = self.bucket_size)

            # gather across batch and divide across world

            (x, mask), batch_sizes, num_sharded_batches = sharded_batch_to_sharded_seq(x, mask, self.ring_seq_size)

            if exists(labels):
                (labels, _), *_ = sharded_batch_to_sharded_seq(labels, None, self.ring_seq_size)

            # calculate ring size from num sharded batches

            ring_size = get_world_size() // num_sharded_batches

        # rotary positions
        # taking into account ring and striping

        rotary_emb = self.rotary_emb(x.shape[-1])

        # main transformer logic

        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(
                x,
                mask = mask,
                rotary_emb = rotary_emb,
                force_ring_reduce_off = force_ring_reduce_off,
                ring_size = ring_size
            ) + x

            x = ff(x) + x

        logits = self.to_logits(x)

        # handle returning of loss

        if return_loss:
            logits = rearrange('b n c -> b c n', logits)

            ce_loss = F.cross_entropy(
                logits,
                labels,
                ignore_index = self.ignore_index
            )

            return ce_loss

        # otherwise gather all sequence chunks for logits across machines and shard the batch dimension

        if not auto_shard_seq:
            return logits

        logits = sharded_seq_to_sharded_batch(logits, batch_sizes, num_sharded_batches)

        if self.striped_ring_attn:
            logits = rearrange('b (i j) d -> b (j i) d', logits, j = self.bucket_size)

        return logits[:, :seq_len]
