# UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models

Created by [Wenliang Zhao](https://wl-zhao.github.io/)\*, [Lujia Bai](https://openreview.net/profile?id=~Lujia_Bai1)*, [Yongming Rao](https://raoyongming.github.io/)\*, [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1)

This code contains the Pytorch implementation for UniPC.

UniPC is a training-free framework designed for the fast sampling of diffusion models, which consists of a corrector (UniC) and a predictor (UniP) that share a unified analytical form and support arbitrary orders.

[[Project Page]](https://unipc.ivg-research.xyz/) [[arXiv]](https://arxiv.org/abs/xxxx.xxxxx)

![intro](assets/intro.png)

UniPC is by desinged model-agnostic, supporting pixel-space/latent-space DPMs on unconditional/conditional sampling. It can also be applied to both noise prediction model and data prediction model. 

Compared with previous methods, UniPC converges faster thanks to the increased order of accuracy. Both quantitative and qualitative results show UniPC can remarkably improve the sampling quality, especially in extreme few steps (5~10).

![demo](assets/demo.png)

# Code Examples
We provide code examples based on the [ScoreSDE](https://github.com/yang-song/score_sde) and [Stable-Diffusion](https://github.com/CompVis/stable-diffusion) in the `example` folder. Please follow the `README.md` file in the corresponding examples for further instructions to use our UniPC.
## ScoreSDE with UniPC
We provide a pytorch example in `example/score_sde_pytorch`, where we show how to use our UniPC to sample from a DPM pre-trained on CIFAR10.

## Stable-Diffusion with UniPC

We provide an example of applying UniPC to stable-diffusion in `example/stable-diffusion`. Our UniPC can accelerate the sampling in both conditional and unconditional sampling.


# Acknowledgement

Our code is based on [ScoreSDE](https://github.com/yang-song/score_sde), [Stable-Diffusion](https://github.com/CompVis/stable-diffusion), and [DPM-Solver](https://github.com/LuChengTHU/dpm-solver).

# Citation

If you find our work useful in your research, please consider citing:

```
@article{zhao2023unipc,
  title={UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models},
  author={Zhao, Wenliang and Bai, Lujia and Rao, Yongming and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:xxxx:xxxxxx},
  year={2023}
}
```
