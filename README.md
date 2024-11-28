# A neural diffusion model for identifying influential nodes in complex networks

 
## NDM
 
The repository contains an implementation of a neural diffusion model (NDM) designed to identify weighted influential nodes in complex networks.

The associated paper to this repository can be found here:
<a href="https://doi.org/10.1016/j.chaos.2024.115682" > A neural diffusion model for identifying influential nodes in complex networks  </a> 

## Abstract
Identifying influential nodes in complex networks through influence diffusion models is a challenging problem
that has garnered significant attention in recent years. While many heuristic algorithms have been developed to
address this issue, neural models that account for weighted influence remain underexplored. In this paper, we
introduce a neural diffusion model (NDM) designed to identify weighted influential nodes in complex networks.
Our NDM is trained on small-scale networks and learns to map network structures to the corresponding
weighted influence of nodes, leveraging the weighted independent cascade model to provide insights into
network dynamics. Specifically, we extract weight-based features from nodes at various scales to capture their
local structures. We then employ a neural encoder to incorporate neighborhood information and learn node
embeddings by integrating features across different scales into sequential neural units. Finally, a decoding
mechanism transforms these node embeddings into estimates of weighted influence. Experimental results on
both real-world and synthetic networks demonstrate that our NDM outperforms state-of-the-art techniques,
achieving superior prediction performance.


## Keywords
 Influential nodes · Complex network · Weighted independent model · Deep learning
 
 
![Overview of the proposed neural diffusion model (NDM) training framework](https://github.com/User2021-ai/NDM/blob/main/Overview%20of%20NDM.pdf)

Overview of the proposed neural diffusion model (NDM) training framework.

# How to Cite
Please cite the following paper:<br>
<a href="https://doi.org/10.1016/j.chaos.2024.115682" > A neural diffusion model for identifying influential nodes in complex networks  </a> 
 


 ```ruby
Ahmad, W., & Wang, B. (2024). A neural diffusion model for identifying influential nodes in complex networks. Chaos, Solitons & Fractals, 189, 115682.

```
BibTeX
```ruby
@article{ahmad2024neural,
  title={A neural diffusion model for identifying influential nodes in complex networks},
  author={Ahmad, Waseem and Wang, Bang},
  journal={Chaos, Solitons \& Fractals},
  volume={189},
  pages={115682},
  year={2024},
  publisher={Elsevier}
}
``` 
 
 
 
 
 



