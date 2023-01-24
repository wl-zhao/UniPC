# UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models

This code contains the Pytorch implementation for UniPC.


UniPC is a new framework designed for the fast sampling of diffusion models, which consists of a corrector (UniC) and a predictor (UniP) that share a unified analytical form and support arbitrary orders. 


To use UniPC to sample from any pre-trained diffusion model, please copy `uni_pc.py` and import UniPC from it. We provide a code example based on the [ScoreSDE](https://github.com/yang-song/score_sde), as in `example/score_sde_pytorch`. Please simply go to the folder and prepare the required data following the instructions in `example/score_sde_pytorch/README.md`, and run

```python
bash sample.sh 0 # using cuda:0
```

# Acknowledgement

Our code is based on [ScoreSDE](https://github.com/yang-song/score_sde) and [DPM-Solver](https://github.com/LuChengTHU/dpm-solver).
