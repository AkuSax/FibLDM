#!/bin/bash

VAE_CHECKPOINT="../model_runs/vae_run_3/vae_best.pth" 

python ../train_ldm_unet.py \
    --latent_data_dir ../data \
    --save_dir ../model_runs/ldm_unet_run_6 \
    --vae_checkpoint $VAE_CHECKPOINT \
    --latent_dim 32 \
    --latent_size 16 \
    --epochs 300 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_workers 4 \
    --noise_steps 1000 \
    --noise_schedule cosine \
    --save_interval 10 