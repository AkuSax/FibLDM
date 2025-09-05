#!/bin/bash

export DATA_DIR="/mnt/hot/public/Yi-Kuan/Fibrosis/Akul/sd_data"
export OUTPUT_DIR="../model_runs/vae_run_4"

mkdir -p $OUTPUT_DIR

accelerate launch --num_processes=2 ../train_vae.py \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --learning_rate=1e-4 \
    --num_train_epochs=100 \
    --train_batch_size=8 \
    --num_dataloader_workers=16 \
    --log_every_epochs=5 \
    --disc_start_epoch=10 \
    --perceptual_weight=0.1 \
    --adversarial_weight=0.05 \
    --subset_fraction=0.01 \
    --log_every_steps=50