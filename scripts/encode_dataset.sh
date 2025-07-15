#!/bin/bash

python ../encode_dataset.py \
  --csv_file ../processed_images/label_processed.csv \
  --img_dir ../processed_images \
  --vae_checkpoint sd_vae \
  --output_dir ../data_sd/ \
  --latent_dim 4 \
  --use_sd_vae