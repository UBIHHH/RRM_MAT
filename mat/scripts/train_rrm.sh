#!/bin/sh
env="RRM"
algo="mat"
exp="rrm_test"
seed=1

echo "env is ${env}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_rrm.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --seed ${seed} \
    --n_training_threads 16 \
    --n_rollout_threads 1 \
    --n_eval_rollout_threads 1 \
    --num_mini_batch 1 \
    --episode_length 25 \
    --num_env_steps 10000000 \
    --ppo_epoch 10 \
    --clip_param 0.05 \
    --lr 5e-4 \
    --critic_lr 5e-4 \
    --use_ReLU \
    --n_block 1 \
    --gain 0.01 \
    --use_eval \
    --eval_interval 10 \
    --n_pbs 3 \
    --n_ues 15 \
    --n_channels 5 \
    --obs_dim 30