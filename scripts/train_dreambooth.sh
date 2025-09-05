#!/bin/bash

export FINETUNED_MODEL_PATH="../model_runs/lora_run_4/best_model"
export VAE_PATH="../model_runs/vae_run_2/best_model_hf"
export CONTROLNET_PATH="../model_runs/lora_run_4/best_model/controlnet"
export INSTANCE_DIR="/mnt/hot/public/Yi-Kuan/Fibrosis/Akul/sd_data/dreambooth_candidates"
export DATA_DIR="/mnt/hot/public/Yi-Kuan/Fibrosis/Akul/sd_data"
export OUTPUT_DIR="../model_runs/dreambooth_fibrosis_run_1"
INSTANCE_PROMPT="a transverse lung CT scan with <fibrosis-texture>"

accelerate launch --mixed_precision="fp16" ../train_dreambooth.py \
  --pretrained_model_path=$FINETUNED_MODEL_PATH/unet_lora \
  --pretrained_vae_path=$VAE_PATH \
  --pretrained_controlnet_path=$CONTROLNET_PATH \
  --instance_data_dir=$INSTANCE_DIR \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="$INSTANCE_PROMPT" \
  --resolution=256 \
  --train_batch_size=16 \
  --num_dataloader_workers=0 \
  --learning_rate=2e-6 \
  --max_train_steps=800 \
  --log_every_steps=100