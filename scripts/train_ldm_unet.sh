#!/bin/bash

VAE_CHECKPOINT="../model_runs/sd_vae/vae_best.pth" 

python ../train_ldm_unet.py \
    --latent_data_dir /hot/Yi-Kuan/Fibrosis/Akul/sd_data/ \
    --save_dir ../model_runs/sd_run_1 \
    --vae_checkpoint $VAE_CHECKPOINT \
    --stats_path /hot/Yi-Kuan/Fibrosis/Akul/sd_data/latent_stats.pt
    --latent_dim 4 \
    --latent_size 32 \
    --epochs 300 \
    --batch_size 96 \
    --lr 3e-5 \
    --num_workers 16 \
    --noise_steps 1000 \
    --noise_schedule cosine \
    --save_interval 10 \
    --lambda_mse 1.0 \
    --lambda_lpips 0.5 \
    --lambda_img_lpips 1.0