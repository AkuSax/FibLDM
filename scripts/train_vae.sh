#!/bin/bash

# This script trains the Variational Autoencoder (VAE) which is the first
# stage of the Latent Diffusion Model pipeline. The trained VAE is used to
# encode the dataset into a latent space and to decode generated latents
# back into full-resolution images.

python ../train_autoencoder.py \
    --csv_file ../small_label.csv \
    --data_dir /hot/Yi-Kuan/Fibrosis/ \
    --save_dir vae_run_1 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --latent_dim 8 \
    --kld_weight 1e-4 