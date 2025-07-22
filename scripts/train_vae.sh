#!/bin/bash

# Directory where your processed PNGs are located
DATA_DIR="../processed_images"

# --- Train the VAE ---
python ../train_autoencoder.py \
    --label_file "$DATA_DIR/clustered_subset.csv" \
    --data_dir "$DATA_DIR" \
    --save_dir ../model_runs/sd_vae \
    --epochs 100 \
    --batch_size 5 \
    --lr 1e-4 \
    --latent_dim 32 \
    --kld_weight 1e-5 \
    --num_workers 16 \
    --use_sd_vae
