#!/bin/bash

python ../interpolate.py \
    --vae_checkpoint ../model_runs/vae_run_2/vae_best.pth \
    --label_file /hot/Yi-Kuan/Fibrosis/label.csv \
    --data_dir /hot/Yi-Kuan/Fibrosis/ \
    --save_dir ../interpolation_samples \
    --latent_dim 32 \
    --idx1 10 \
    --idx2 50 \
    --steps 8