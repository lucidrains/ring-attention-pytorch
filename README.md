<img src="./ring.png" width="450px"></img>

## Ring Attention - Pytorch

Implementation of <a href="https://arxiv.org/abs/2310.01889">Ring Attention</a>, from <a href="https://www.haoliu.site/">Liu</a> et al. at Berkeley AI, in Pytorch.

It basically splits the data across the sequence dimension (instead of batch) and applies ring reduce to the processing of the tiles of the attention matrix, flash attention style.

I believe this is being used for the 1-10 million tokens for the latest Gemini. At least some form of it; the other possibility would be unpublished improvements on top of <a href="https://github.com/lucidrains/recurrent-memory-transformer-pytorch">RMT</a>.

In addition, the repository also contains the logic for <a href="https://arxiv.org/abs/2311.09431">Striped Attention</a>, a follow up paper that permutes the sequence for better workload balancing for autoregressive transformers.

It also contains support for <a href="https://arxiv.org/abs/2305.13245">grouped query attention</a>, popularized by Llama series of attention models. This will further save on communication costs during the ring reduce.

## Appreciation

- <a href="https://a16z.com/supporting-the-open-source-ai-community/">A16Z Open Source AI Grant Program</a> for the generous sponsorship, as well as my other sponsors, for affording me the independence to open source current artificial intelligence research

- <a href="https://tridao.me/">Tri Dao</a> for all his tremendous hard work maintaining <a href="https://github.com/Dao-AILab/flash-attention">Flash Attention</a> over the last year or two, from which the CUDA version in this repository depends on

- Phil Tillet for <a href="https://github.com/openai/triton">Triton</a>, without which the forward ring flash attention CUDA kernel would have taken a magnitude of order more work.

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

This repository also contains an implementation of <a href="https://arxiv.org/abs/2408.04093">Tree Attention Decoding</a> from Shyam et al.

It can be imported and used as follows

```python
from ring_attention_pytorch import tree_attn_decode

out = tree_attn_decode(q, k, v) # where q, k, v exists across all machines
```

## Test

First install requirements

```bash
$ pip install -r requirements.txt
```

Then say testing autoregressive striped ring attention on cuda would be

```bash
$ python assert.py --use-cuda --causal --striped-ring-attn
```

Testing tree attention would be

```bash
$ python assert_tree_attn.py --use-cuda --seq-len 8192
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
- [x] add flash attention kernel version in the presence of cuda
    - [x] for forwards, use modified Triton flash attention forwards that outputs row sums, maxes, and exponentiated weighted sum
    - [x] for backwards, use Tri's flash attention kernels, accumulate dq, dk, dv across rings
    - [x] refactor to have naive ring+flash attention work with `(batch, seq, head, dim)`
    - [x] handle key padding mask for forwards by translating mask to bias
    - [x] figure out how Tri handles key padding mask for backwards
    - [x] scale output of flash attention forwards on the last ring pass reduce
    - [x] verify backwards working in a100 runpod
    - [x] dk, dv needs to be float32, while kv needs to be float16. see if both can be cast to int before stacked and ring passed all in one go, then reinterpret back to float32 and float16
    - [x] prevent an unnecessary `tl.load` on the first ring pass
    - [x] cuda backwards pass must have same dq, dk, dv as naive
- [x] fix naive flash attention backwards
- [x] validate cuda causal and striped ring attention works
- [x] make sure cuda striped attention works for multiple buckets, otherwise flash attention is ineffective
- [x] for cuda striped attention, for backwards hack, pad the extra token once and index out when passing into Tri's cuda kernel
- [x] find a machine with 8 GPUs and test with a quarter million tokens first
- [x] see for cuda version whether softmax_D can be computed once and cached over the ring reduce. go for modified triton backwards if not

- [ ] think about how to craft a special `Dataset` that shards across sequence length (take into account labels for cross entropy loss) for ring transformer training
- [ ] add ring attention to Tri's flash attention implementation. find some cuda ring reduce impl
- [ ] figure out how to pytest distributed pytorch
- [ ] use sdp context manager to validate when it is possible to use `ring_flash_attn_cuda`, otherwise assert out
- [ ] improvise a variant where each machine keeps compressed summary tokens, and one only ring pass those summary token for some given distance

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

```bibtex
@article{Tillet2019TritonAI,
    title   = {Triton: an intermediate language and compiler for tiled neural network computations},
    author  = {Philippe Tillet and H. Kung and D. Cox},
    journal = {Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages},
    year    = {2019}
}
```

```bibtex
@article{Ainslie2023GQATG,
    title   = {GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
    author  = {Joshua Ainslie and James Lee-Thorp and Michiel de Jong and Yury Zemlyanskiy and Federico Lebr'on and Sumit K. Sanghai},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2305.13245},
    url     = {https://api.semanticscholar.org/CorpusID:258833177}
}
```

```bibtex
@inproceedings{Shyam2024TreeAT,
    title   = {Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters},
    author  = {Vasudev Shyam and Jonathan Pilault and Emily Shepperd and Quentin Anthony and Beren Millidge},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:271769078}
}
```

```bibtex
@article{Dubey2024TheL3,
    title   = {The Llama 3 Herd of Models},
    author  = {Abhimanyu Dubey and Abhinav Jauhri and Abhinav Pandey and Abhishek Kadian and Ahmad Al-Dahle and Aiesha Letman and Akhil Mathur and Alan Schelten and Amy Yang and Angela Fan and Anirudh Goyal and Anthony Hartshorn and Aobo Yang and Archi Mitra and Archie Sravankumar and Artem Korenev and Arthur Hinsvark and Arun Rao and Aston Zhang and Aurelien Rodriguez and Austen Gregerson and Ava Spataru and Baptiste Rozi{\`e}re and Bethany Biron and Binh Tang and Bobbie Chern and Charlotte Caucheteux and Chaya Nayak and Chloe Bi and Chris Marra and Chris McConnell and Christian Keller and Christophe Touret and Chunyang Wu and Corinne Wong and Cristian Cant{\'o}n Ferrer and Cyrus Nikolaidis and Damien Allonsius and Daniel Song and Danielle Pintz and Danny Livshits and David Esiobu and Dhruv Choudhary and Dhruv Mahajan and Diego Garcia-Olano and Diego Perino and Dieuwke Hupkes and Egor Lakomkin and Ehab A. AlBadawy and Elina Lobanova and Emily Dinan and Eric Michael Smith and Filip Radenovic and Frank Zhang and Gabriele Synnaeve and Gabrielle Lee and Georgia Lewis Anderson and Graeme Nail and Gr{\'e}goire Mialon and Guanglong Pang and Guillem Cucurell and Hailey Nguyen and Hannah Korevaar and Hu Xu and Hugo Touvron and Iliyan Zarov and Imanol Arrieta Ibarra and Isabel M. Kloumann and Ishan Misra and Ivan Evtimov and Jade Copet and Jaewon Lee and Jan Laurens Geffert and Jana Vranes and Jason Park and Jay Mahadeokar and Jeet Shah and Jelmer van der Linde and Jennifer Billock and Jenny Hong and Jenya Lee and Jeremy Fu and Jianfeng Chi and Jianyu Huang and Jiawen Liu and Jie Wang and Jiecao Yu and Joanna Bitton and Joe Spisak and Jongsoo Park and Joseph Rocca and Joshua Johnstun and Joshua Saxe and Ju-Qing Jia and Kalyan Vasuden Alwala and K. Upasani and Kate Plawiak and Keqian Li and Ken-591 neth Heafield and Kevin Stone and Khalid El-Arini and Krithika Iyer and Kshitiz Malik and Kuenley Chiu and Kunal Bhalla and Lauren Rantala-Yeary and Laurens van der Maaten and Lawrence Chen and Liang Tan and Liz Jenkins and Louis Martin and Lovish Madaan and Lubo Malo and Lukas Blecher and Lukas Landzaat and Luke de Oliveira and Madeline C. Muzzi and Mahesh Babu Pasupuleti and Mannat Singh and Manohar Paluri and Marcin Kardas and Mathew Oldham and Mathieu Rita and Maya Pavlova and Melissa Hall Melanie Kambadur and Mike Lewis and Min Si and Mitesh Kumar Singh and Mona Hassan and Naman Goyal and Narjes Torabi and Nikolay Bashlykov and Nikolay Bogoychev and Niladri S. Chatterji and Olivier Duchenne and Onur cCelebi and Patrick Alrassy and Pengchuan Zhang and Pengwei Li and Petar Vasi{\'c} and Peter Weng and Prajjwal Bhargava and Pratik Dubal and Praveen Krishnan and Punit Singh Koura and Puxin Xu and Qing He and Qingxiao Dong and Ragavan Srinivasan and Raj Ganapathy and Ramon Calderer and Ricardo Silveira Cabral and Robert Stojnic and Roberta Raileanu and Rohit Girdhar and Rohit Patel and Romain Sauvestre and Ronnie Polidoro and Roshan Sumbaly and Ross Taylor and Ruan Silva and Rui Hou and Rui Wang and Saghar Hosseini and Sahana Chennabasappa and Sanjay Singh and Sean Bell and Seohyun Sonia Kim and Sergey Edunov and Shaoliang Nie and Sharan Narang and Sharath Chandra Raparthy and Sheng Shen and Shengye Wan and Shruti Bhosale and Shun Zhang and Simon Vandenhende and Soumya Batra and Spencer Whitman and Sten Sootla and Stephane Collot and Suchin Gururangan and Sydney Borodinsky and Tamar Herman and Tara Fowler and Tarek Sheasha and Thomas Georgiou and Thomas Scialom and Tobias Speckbacher and Todor Mihaylov and Tong Xiao and Ujjwal Karn and Vedanuj Goswami and Vibhor Gupta and Vignesh Ramanathan and Viktor Kerkez and Vincent Gonguet and Virginie Do and Vish Vogeti and Vladan Petrovic and Weiwei Chu and Wenhan Xiong and Wenyin Fu and Whitney Meers and Xavier Martinet and Xiaodong Wang and Xiaoqing Ellen Tan and Xinfeng Xie and Xuchao Jia and Xuewei Wang and Yaelle Goldschlag and Yashesh Gaur and Yasmine Babaei and Yiqian Wen and Yiwen Song and Yuchen Zhang and Yue Li and Yuning Mao and Zacharie Delpierre Coudert and Zhengxu Yan and Zhengxing Chen and Zoe Papakipos and Aaditya K. Singh and Aaron Grattafiori and Abha Jain and Adam Kelsey and Adam Shajnfeld and Adi Gangidi and Adolfo Victoria and Ahuva Goldstand and Ajay Menon and Ajay Sharma and Alex Boesenberg and Alex Vaughan and Alexei Baevski and Allie Feinstein and Amanda Kallet and Amit Sangani and Anam Yunus and Andrei Lupu and Andres Alvarado and Andrew Caples and Andrew Gu and Andrew Ho and Andrew Poulton and Andrew Ryan and Ankit Ramchandani and Annie Franco and Aparajita Saraf and Arkabandhu Chowdhury and Ashley Gabriel and Ashwin Bharambe and Assaf Eisenman and Azadeh Yazdan and Beau James and Ben Maurer and Ben Leonhardi and Bernie Huang and Beth Loyd and Beto De Paola and Bhargavi Paranjape and Bing Liu and Bo Wu and Boyu Ni and Braden Hancock and Bram Wasti and Brandon Spence and Brani Stojkovic and Brian Gamido and Britt Montalvo and Carl Parker and Carly Burton and Catalina Mejia and Changhan Wang and Changkyu Kim and Chao Zhou and Chester Hu and Ching-Hsiang Chu and Chris Cai and Chris Tindal and Christoph Feichtenhofer and Damon Civin and Dana Beaty and Daniel Kreymer and Shang-Wen Li and Danny Wyatt and David Adkins and David Xu and Davide Testuggine and Delia David and Devi Parikh and Diana Liskovich and Didem Foss and Dingkang Wang and Duc Le and Dustin Holland and Edward Dowling and Eissa Jamil and Elaine Montgomery and Eleonora Presani and Emily Hahn and Emily Wood and Erik Brinkman and Esteban Arcaute and Evan Dunbar and Evan Smothers and Fei Sun and Felix Kreuk and Feng Tian and Firat Ozgenel and Francesco Caggioni and Francisco Guzm'an and Frank J. Kanayet and Frank Seide and Gabriela Medina Florez and Gabriella Schwarz and Gada Badeer and Georgia Swee and Gil Halpern and Govind Thattai and Grant Herman and Grigory G. Sizov and Guangyi Zhang and Guna Lakshminarayanan and Hamid Shojanazeri and Han Zou and Hannah Wang and Han Zha and Haroun Habeeb and Harrison Rudolph and Helen Suk and Henry Aspegren and Hunter Goldman and Igor Molybog and Igor Tufanov and Irina-Elena Veliche and Itai Gat and Jake Weissman and James Geboski and James Kohli and Japhet Asher and Jean-Baptiste Gaya and Jeff Marcus and Jeff Tang and Jennifer Chan and Jenny Zhen and Jeremy Reizenstein and Jeremy Teboul and Jessica Zhong and Jian Jin and Jingyi Yang and Joe Cummings and Jon Carvill and Jon Shepard and Jonathan McPhie and Jonathan Torres and Josh Ginsburg and Junjie Wang and Kaixing(Kai) Wu and U KamHou and Karan Saxena and Karthik Prasad and Kartikay Khandelwal and Katayoun Zand and Kathy Matosich and Kaushik Veeraraghavan and Kelly Michelena and Keqian Li and Kun Huang and Kunal Chawla and Kushal Lakhotia and Kyle Huang and Lailin Chen and Lakshya Garg and A Lavender and Leandro Silva and Lee Bell and Lei Zhang and Liangpeng Guo and Licheng Yu and Liron Moshkovich and Luca Wehrstedt and Madian Khabsa and Manav Avalani and Manish Bhatt and Maria Tsimpoukelli and Martynas Mankus and Matan Hasson and Matthew Lennie and Matthias Reso and Maxim Groshev and Maxim Naumov and Maya Lathi and Meghan Keneally and Michael L. Seltzer and Michal Valko and Michelle Restrepo and Mihir Patel and Mik Vyatskov and Mikayel Samvelyan and Mike Clark and Mike Macey and Mike Wang and Miquel Jubert Hermoso and Mo Metanat and Mohammad Rastegari and Munish Bansal and Nandhini Santhanam and Natascha Parks and Natasha White and Navyata Bawa and Nayan Singhal and Nick Egebo and Nicolas Usunier and Nikolay Pavlovich Laptev and Ning Dong and Ning Zhang and Norman Cheng and Oleg Chernoguz and Olivia Hart and Omkar Salpekar and Ozlem Kalinli and Parkin Kent and Parth Parekh and Paul Saab and Pavan Balaji and Pedro Rittner and Philip Bontrager and Pierre Roux and Piotr Doll{\'a}r and Polina Zvyagina and Prashant Ratanchandani and Pritish Yuvraj and Qian Liang and Rachad Alao and Rachel Rodriguez and Rafi Ayub and Raghotham Murthy and Raghu Nayani and Rahul Mitra and Raymond Li and Rebekkah Hogan and Robin Battey and Rocky Wang and Rohan Maheswari and Russ Howes and Ruty Rinott and Sai Jayesh Bondu and Samyak Datta and Sara Chugh and Sara Hunt and Sargun Dhillon and Sasha Sidorov and Satadru Pan and Saurabh Verma and Seiji Yamamoto and Sharadh Ramaswamy and Shaun Lindsay and Sheng Feng and Shenghao Lin and Shengxin Cindy Zha and Shiva Shankar and Shuqiang Zhang and Sinong Wang and Sneha Agarwal and Soji Sajuyigbe and Soumith Chintala and Stephanie Max and Stephen Chen and Steve Kehoe and Steve Satterfield and Sudarshan Govindaprasad and Sumit Gupta and Sung-Bae Cho and Sunny Virk and Suraj Subramanian and Sy Choudhury and Sydney Goldman and Tal Remez and Tamar Glaser and Tamara Best and Thilo Kohler and Thomas Robinson and Tianhe Li and Tianjun Zhang and Tim Matthews and Timothy Chou and Tzook Shaked and Varun Vontimitta and Victoria Ajayi and Victoria Montanez and Vijai Mohan and Vinay Satish Kumar and Vishal Mangla and Vlad Ionescu and Vlad Andrei Poenaru and Vlad T. Mihailescu and Vladimir Ivanov and Wei Li and Wenchen Wang and Wenwen Jiang and Wes Bouaziz and Will Constable and Xia Tang and Xiaofang Wang and Xiaojian Wu and Xiaolan Wang and Xide Xia and Xilun Wu and Xinbo Gao and Yanjun Chen and Ye Hu and Ye Jia and Ye Qi and Yenda Li and Yilin Zhang and Ying Zhang and Yossi Adi and Youngjin Nam and Yu Wang and Yuchen Hao and Yundi Qian and Yuzi He and Zach Rait and Zachary DeVito and Zef Rosnbrick and Zhaoduo Wen and Zhenyu Yang and Zhiwei Zhao},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2407.21783},
    url     = {https://api.semanticscholar.org/CorpusID:271571434}
}
```

*<a href="http://www.incompleteideas.net/IncIdeas/BitterLesson.html">The Bitter Lesson</a>* - Richard Sutton
