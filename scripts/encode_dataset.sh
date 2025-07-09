#!/bin/bash

python ../encode_dataset.py \
  --csv_file ../processed_images/label_processed.csv \
  --img_dir ../processed_images \
  --vae_checkpoint ../model_runs/vae_run_processed/vae_best.pth \
  --output_dir ../data/ \
  --latent_dim 32