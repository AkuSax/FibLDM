#!/bin/bash

export DATA_DIR="/hot/Yi-Kuan/Fibrosis/Akul/sd_data"
export OUTPUT_DIR="../model_runs/lora_run/lora_run_4"

LORA_RANK=32
LEARNING_RATE=5e-5
NUM_WORKERS=30
NUM_EPOCHS=150
BATCH_SIZE=32
EVAL_EPOCHS=5
EARLY_STOPPING_PATIENCE=15

mkdir -p $OUTPUT_DIR

python ../train_lora_unet.py \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --lora_rank=$LORA_RANK \
    --learning_rate=$LEARNING_RATE \
    --num_train_epochs=$NUM_EPOCHS \
    --train_batch_size=$BATCH_SIZE \
    --num_dataloader_workers=$NUM_WORKERS \
    --eval_epochs=$EVAL_EPOCHS \
    --early_stopping_patience=$EARLY_STOPPING_PATIENCE \