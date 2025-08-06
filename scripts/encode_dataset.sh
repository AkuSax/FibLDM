#!/bin/bash

export DATA_DIR="/hot/Yi-Kuan/Fibrosis/Akul/sd_data"
export VAE_PATH="../model_runs/vae_run_2/best_model_hf"
export OUTPUT_DIR=$DATA_DIR

python ../encode_dataset.py \
    --vae_model_path=$VAE_PATH \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --batch_size=128