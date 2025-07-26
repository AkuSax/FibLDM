#!/bin/bash

python ../encode_dataset.py \
  --csv_file ../processed_images/label.csv \
  --img_dir ../processed_images \
  --vae_checkpoint ../model_runs/sd_vae/vae_best.pth \
  --output_dir /hot/Yi-Kuan/Fibrosis/Akul/sd_data \
  --latent_dim 4 \
  --use_sd_vae 