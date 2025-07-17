#!/bin/bash

# Directory where your processed PNGs are located
DATA_DIR="../processed_images_512"

# --- Train the VAE ---
python ../train_autoencoder.py \
    --label_file "$DATA_DIR/label.csv" \
    --data_dir "$DATA_DIR" \
    --save_dir ../model_runs/sd_vae \
    --epochs 20 \
    --batch_size 6 \
    --lr 1e-4 \
    --latent_dim 32 \
    --kld_weight 1e-5 \
    --num_workers 16 \
    --use_sd_vae
    
echo "VAE training complete"
