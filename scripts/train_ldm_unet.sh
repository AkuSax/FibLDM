#!/bin/bash

VAE_CHECKPOINT="../model_runs/vae_run_processed/vae_best.pth" 
STATS_PATH="../data/latent_stats.pt"

python ../train_ldm_unet.py \
    --latent_data_dir ../data \
    --save_dir ../model_runs/ldm_unet_run_10 \
    --vae_checkpoint $VAE_CHECKPOINT \
    --stats_path $STATS_PATH \
    --latent_dim 32 \
    --latent_size 16 \
    --epochs 300 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_workers 4 \
    --noise_steps 1000 \
    --noise_schedule cosine \
    --save_interval 10 