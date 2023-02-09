# UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models

Created by [Wenliang Zhao](https://wl-zhao.github.io/)\*, [Lujia Bai](https://openreview.net/profile?id=~Lujia_Bai1)*, [Yongming Rao](https://raoyongming.github.io/)\*, [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1)

This code contains the Pytorch implementation for UniPC.

UniPC is a training-free framework designed for the fast sampling of diffusion models, which consists of a corrector (UniC) and a predictor (UniP) that share a unified analytical form and support arbitrary orders.

[[Project Page]](https://unipc.ivg-research.xyz/) [[arXiv]](https://arxiv.org/abs/xxxx.xxxxx)

![intro](assets/intro.png)

UniPC is by desinged model-agnostic, supporting pixel-space/latent-space DPMs on unconditional/conditional sampling. It can also be applied to both noise prediction modle and data prediction model. 

# Examples

![demo](assets/demo.png)

We provide code examples based on the [ScoreSDE](https://github.com/yang-song/score_sde) and [Stable-Diffusion](https://github.com/CompVis/stable-diffusion) in the `example` folder. Please follow the `README.md` file in the corresponding examples for further instructions to use our UniPC.

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
