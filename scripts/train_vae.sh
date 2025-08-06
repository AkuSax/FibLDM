#!/bin/bash

export DATA_DIR="/hot/Yi-Kuan/Fibrosis/Akul/sd_data"
export OUTPUT_DIR="../model_runs/vae_run_2"

accelerate launch --num_processes=2 --mixed_precision="fp16" ../train_vae.py \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --learning_rate=2e-4 \
    --num_train_epochs=50 \
    --train_batch_size=16 \
    --val_batch_size=16 \
    --num_dataloader_workers=30 \
    --log_every_epochs=2 \
    --early_stopping_patience=10 \
    --subset_fraction=0.01 \
    --log_every_steps=10