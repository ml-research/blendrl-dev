export CUDA_VISIBLE_DEVICES=10
nohup python train_blenderl.py --env-name kangaroo --track --joint-training --num-envs 512 --num-steps 126 --seed 0 > logs/gpu_10.log&
export CUDA_VISIBLE_DEVICES=11
nohup python train_blenderl.py --env-name kangaroo --track --joint-training --num-envs 512 --num-steps 126 --seed 512 > logs/gpu_11.log &
export CUDA_VISIBLE_DEVICES=12
nohup python train_blenderl.py --env-name kangaroo --track --joint-training --num-envs 512 --num-steps 126 --seed 1024 > logs/gpu_12.log &


export CUDA_VISIBLE_DEVICES=13
nohup python train_blenderl.py --env-name seaquest --track --joint-training --num-envs 512 --num-steps 126 --reasoner neumann --seed 0 > logs/gpu_13.log&
export CUDA_VISIBLE_DEVICES=14
nohup python train_blenderl.py --env-name seaquest --track --joint-training --num-envs 512 --num-steps 126 --reasoner neumann --seed 512 > logs/gpu_14.log &
export CUDA_VISIBLE_DEVICES=15
nohup python train_blenderl.py --env-name seaquest --track --joint-training --num-envs 512 --num-steps 126 --reasoner neumann --seed 1024 > logs/gpu_15.log &