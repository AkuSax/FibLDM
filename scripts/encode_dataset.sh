#!/bin/bash

python ../encode_dataset.py \
  --csv_file ./data/small_label.csv \
  --img_dir ./data/ \
  --vae_checkpoint ../model_runs/vae_run_1/vae_best.pth \
  --output_dir ../data/latents_dataset \
  --latent_dim 8