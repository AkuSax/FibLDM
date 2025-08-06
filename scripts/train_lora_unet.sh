#!/bin/bash

export DATA_DIR="/hot/Yi-Kuan/Fibrosis/Akul/sd_data"
export VAE_PATH="../model_runs/vae_run_2/best_model_hf" 
export OUTPUT_DIR="../model_runs/lora_run_1"

mkdir -p $OUTPUT_DIR

accelerate launch --num_processes=2 --mixed_precision="fp16" ../train_lora_unet.py \
    --data_dir=$DATA_DIR \
    --vae_model_path=$VAE_PATH \
    --output_dir=$OUTPUT_DIR \
    --lora_rank=32 \
    --learning_rate=1e-4 \
    --num_train_epochs=100 \
    --train_batch_size=16 \
    --num_dataloader_workers=16 \
    --log_every_epochs=5