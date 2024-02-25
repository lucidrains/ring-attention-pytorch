<img src="./ring.png" width="450px"></img>

## Ring Attention - Pytorch (wip)

Explorations into <a href="https://arxiv.org/abs/2310.01889">Ring Attention</a>, from <a href="https://www.haoliu.site/">Liu</a> et al. at Berkeley AI.

It basically splits the data across the sequence dimension (instead of batch) and applies ring reduce to the processing of the tiles of the attention matrix, flash attention style.

I believe this is being used for the 1-10 million tokens for the latest Gemini. At least some form of it; the other possibility would be unpublished improvements on top of <a href="https://github.com/lucidrains/recurrent-memory-transformer-pytorch">RMT</a>.

In addition, the repository also contains the logic for <a href="https://arxiv.org/abs/2311.09431">Striped Attention</a>, a follow up paper that permutes the sequence for better workload balancing for autoregressive transformers.

## Appreciation

- <a href="https://a16z.com/supporting-the-open-source-ai-community/">A16Z Open Source AI Grant Program</a> for the generous sponsorship, as well as my other sponsors, for affording me the independence to open source current artificial intelligence research

## Install

```bash
$ pip install ring-attention-pytorch
```

## Usage

```python
import torch
from ring_attention_pytorch import RingAttention

attn = RingAttention(
    dim = 512,
    dim_head = 64,
    heads = 8,
    causal = True,
    auto_shard_seq = True,
    ring_attn = True,
    ring_seq_size = 512
)

tokens = torch.randn(1, 1024, 512)
attended = attn(tokens)

assert attended.shape == tokens.shape
```

## Test

```bash
$ python assert.py
```

## Todo

- [x] make it work with derived causal mask based on rank and chunk sizes
- [x] modify flash attention to output intermediates and figure out backwards with recompute and ring passes
- [x] functions for splitting the sequence evenly among ranks, either within attention function, or in the external ring transformer wrapper
- [x] basic test case with two processes and check for equivalent output and gradients
- [x] testing
    - [x] make sure key padding mask works
    - [x] make sure causal mask works
    - [x] rotary embeddings, with proper key/value offset depending on ring rank
- [x] striped attention
    - [x] add the permutating logic before and after transformer
    - [x] add causal masking logic - account for sub bucketing by flash attention
- [x] fix issue with ring attention when flash buckets > 1
- [x] move flash attention back to key / value column traversal on outer loop and save on ring communication
    - [x] backwards
    - [x] forwards
- [x] fix rotary positions for striped ring attention when flash buckets > 1
- [x] allow for variable ring passes per layer, for <a href="https://arxiv.org/abs/2007.03356">local -> global attention</a> in ring transformer as one goes up the layers.
- [x] when doing ring passes, alternate between designated send and receive buffers
- [x] instead of max ring passes, able to specify lookback in terms of sequence length, and derive number of flash attention bucket + ring passes from that
- [x] ability to have ring size < world size, sharding the batch and sequence, and doing ring reduce with the correct set of ranks

- [ ] add flash attention kernel version in the presence of cuda
    - [x] for backwards, use Tri's flash attention kernels, accumulate dq, dk, dv across rings
    - [ ] figure out how Tri handles key padding mask for backwards
    - [ ] for forwards, use modified Triton flash attention forwards that outputs row sums, maxes, and exponentiated weighted sum
- [ ] find a machine with 8 GPUs and test with a quarter million tokens first
- [ ] think about how to craft a special `Dataset` that shards across sequence length (take into account labels for cross entropy loss) for ring transformer training
- [ ] add ring attention to Tri's flash attention implementation. find some cuda ring reduce impl
- [ ] `batch_isend_irecv` in the presence of key padding mask needing ring exchange, but not a big priority
- [ ] figure out how to pytest distributed pytorch

## Citations

```bibtex
@article{Liu2023RingAW,
    title    = {Ring Attention with Blockwise Transformers for Near-Infinite Context},
    author   = {Hao Liu and Matei Zaharia and Pieter Abbeel},
    journal  = {ArXiv},
    year     = {2023},
    volume   = {abs/2310.01889},
    url      = {https://api.semanticscholar.org/CorpusID:263608461}
}
```

```bibtex
@article{Brandon2023StripedAF,
    title   = {Striped Attention: Faster Ring Attention for Causal Transformers},
    author  = {William Brandon and Aniruddha Nrusimha and Kevin Qian and Zachary Ankner and Tian Jin and Zhiye Song and Jonathan Ragan-Kelley},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2311.09431},
    url     = {https://api.semanticscholar.org/CorpusID:265220849}
}
```

```bibtex
@article{Dao2022FlashAttentionFA,
    title   = {FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
    author  = {Tri Dao and Daniel Y. Fu and Stefano Ermon and Atri Rudra and Christopher R'e},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2205.14135}
}
```

```bibtex
@article{dao2023flashattention2,
    title   = {Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning,
    author  = {Dao, Tri},
    year    = {2023}
}
```
