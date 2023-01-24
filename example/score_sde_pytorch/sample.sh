devices=$1
order=3
eps="1e-3"
skip="logSNR"
method="multistep"
sampling_method="uni_pc"
dir="experiments/cifar10_ddpmpp_deep_continuous_steps"

steps=10
eval_folder="uni_pc"_"$method"_order"$order"_step"$steps"
echo $eval_folder

CUDA_VISIBLE_DEVICES=$devices python main.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" --mode "eval" --workdir $dir --config.sampling.eps=$eps --config.sampling.method=$sampling_method --config.sampling.steps=$steps --config.sampling.skip_type=$skip --config.sampling.uni_pc_order=$order --config.sampling.uni_pc_method=$method --config.eval.batch_size=1000 --eval_folder $eval_folder --config.sampling.algorithm_type='data_prediction' --config.sampling.variant='bh1' --config.sampling.lower_order_final=True --config.eval.begin_ckpt 8