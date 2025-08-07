#!/bin/bash

export DATA_DIR="/hot/Yi-Kuan/Fibrosis/Akul/sd_data"
export VAE_PATH="../model_runs/vae_run_2/best_model_hf" 
export OUTPUT_DIR="../model_runs/lora_run_3"

mkdir -p $OUTPUT_DIR

accelerate launch --num_processes=2 --mixed_precision="fp16" ../train_controlnet_lora.py \
    --data_dir=$DATA_DIR \
    --vae_model_path=$VAE_PATH \
    --output_dir=$OUTPUT_DIR \
    --lora_rank=32 \
    --learning_rate=5e-5 \
    --num_train_epochs=50 \
    --train_batch_size=16 \
    --num_dataloader_workers=38