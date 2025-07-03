#!/bin/bash

python ../encode_dataset.py \
  --csv_file /hot/Yi-Kuan/Fibrosis/label.csv \
  --img_dir /hot/Yi-Kuan/Fibrosis/ \
  --vae_checkpoint ../model_runs/vae_run_3/vae_best.pth \
  --output_dir ../data/ \
  --latent_dim 32