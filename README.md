## Ring Attention - Pytorch (wip)

Explorations into <a href="https://arxiv.org/abs/2310.01889">Ring Attention</a>, from Liu et al. at Berkeley AI

It basically splits the data across the sequence dimension (instead of batch) and applies ring reduce to the processing of the tiles of the attention matrix, flash attention style.

I believe this is being used for the 1-10 million tokens for the latest Gemini. At least some form of it.

## Appreciation

- <a href="https://a16z.com/supporting-the-open-source-ai-community/">A16Z Open Source AI Grant Program</a> for the generous sponsorship, as well as my other sponsors, for affording me the independence to open source current artificial intelligence research

## Todo

- [ ] functions for splitting the sequence evenly among ranks, either within attention function, or in the external ring transformer wrapper
- [ ] basic test case with rank of 1 and 4 and check for equivalent output
- [ ] make it work with derived causal mask based on rank and chunk sizes
- [ ] modify flash attention to output intermediates and figure out backwards with recompute and ring passes
- [ ] figure out striped attention
- [ ] figure out batch_isend_irecv
- [ ] cross attention

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
  title     = {Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning,
  author    = {Dao, Tri},
  year      = {2023}
}
```
