#!/bin/bash

# This script trains the Variational Autoencoder (VAE) which is the first
# stage of the Latent Diffusion Model pipeline. The trained VAE is used to
# encode the dataset into a latent space and to decode generated latents
# back into full-resolution images.

# Set a default data directory
DEFAULT_DATA_DIR="/hot/Yi-Kuan/Fibrosis"
# Use the first argument as the data directory, or the default if not provided
DATA_DIR="${1:-$DEFAULT_DATA_DIR}"

echo "Using data directory: $DATA_DIR"

python ../train_autoencoder.py \
    --label_file "${DATA_DIR}/label.csv" \
    --data_dir "$DATA_DIR" \
    --save_dir ../model_runs/vae_run_3 \
    --epochs 150 \
    --batch_size 32 \
    --lr 1e-4 \
    --latent_dim 32 \
    --kld_weight 1e-3