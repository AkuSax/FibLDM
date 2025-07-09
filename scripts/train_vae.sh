#!/bin/bash

DATA_DIR="../processed_images"

echo "Using data directory: $DATA_DIR"

python ../train_autoencoder.py \
    --label_file "${DATA_DIR}/label_processed.csv" \
    --data_dir "$DATA_DIR" \
    --save_dir ../model_runs/vae_run_processed \
    --epochs 150 \
    --batch_size 32 \
    --lr 1e-4 \
    --latent_dim 32 \
    --kld_weight 1e-3