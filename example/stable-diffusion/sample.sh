CKPT=models/ldm/ffhq256/model.ckpt
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 scripts/sample_diffusion_ddp.py -r $CKPT \
	-n 50000 -l samples/ffhq/uni_pc ${@:2} -c 10 --uni_pc --batch_size 20
