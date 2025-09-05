#!/bin/bash

export FINETUNED_MODEL_PATH="../model_runs/lora_run_4/best_model"
export VAE_PATH="../model_runs/vae_run_2/best_model_hf"
export CONTROLNET_PATH="../model_runs/lora_run_4/best_model/controlnet"
export INSTANCE_DIR="/mnt/hot/public/Yi-Kuan/Fibrosis/Akul/sd_data/dreambooth_candidates"
export DATA_DIR="/mnt/hot/public/Yi-Kuan/Fibrosis/Akul/sd_data"
export OUTPUT_DIR="../model_runs/dreambooth_fibrosis_run_1" 
INSTANCE_PROMPT="a transverse lung CT scan with <fibrosis-texture>"

# Reverted to your original training length and logging frequency.
MAX_STEPS=800
LOG_STEPS=100

echo "Starting FULL training run for ${MAX_STEPS} steps, logging every ${LOG_STEPS} steps."

accelerate launch --num_processes=2 --mixed_precision="fp16" ../train_dreambooth.py \
  --pretrained_model_path=$FINETUNED_MODEL_PATH/unet_lora \
  --pretrained_vae_path=$VAE_PATH \
  --pretrained_controlnet_path=$CONTROLNET_PATH \
  --instance_data_dir=$INSTANCE_DIR \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="$INSTANCE_PROMPT" \
  --resolution=512 \
  --train_batch_size=8 \
  --num_dataloader_workers=0 \
  --learning_rate=2e-6 \
  --max_train_steps=$MAX_STEPS \
  --log_every_steps=$LOG_STEPS