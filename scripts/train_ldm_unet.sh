#!/bin/bash

VAE_CHECKPOINT="../model_runs/vae_run_6/vae_best.pth" 
STATS_PATH="../data/latent_stats.pt"

python ../train_ldm_unet.py \
    --latent_data_dir ../data_sd \
    --save_dir ../model_runs/ldm_unet_run_17 \
    --vae_checkpoint $VAE_CHECKPOINT \
    --stats_path ../data_sd/latent_stats.pt \
    --latent_dim 4 \
    --latent_size 32 \
    --epochs 300 \
    --batch_size 96 \
    --lr 3e-5 \
    --num_workers 16 \
    --noise_steps 1000 \
    --noise_schedule cosine \
    --save_interval 1 \
    --lambda_mse 1.0 \
    --lambda_lpips 0.5 \
    --lambda_img_lpips 1.0 \
    --use_sd_vae