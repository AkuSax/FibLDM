#!/bin/bash

# ControlNet Training Script
# This script trains a ControlNet on top of your pre-trained VAE and LDM

# Set the number of GPUs to use
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1

# Paths to your pre-trained models
VAE_CHECKPOINT="../model_runs/vae_run_3/vae_best.pth"
UNET_CHECKPOINT="../model_runs/ldm_unet_run_1/unet_best.pth" 

# Data paths
DATA_PATH="/hot/Yi-Kuan/Fibrosis"
CSV_PATH="/hot/Yi-Kuan/Fibrosis/label.csv"

# Output directory
SAVE_DIR="../model_runs/controlnet_run_6"

# Training parameters
BATCH_SIZE=64 
NUM_EPOCHS=100
LEARNING_RATE=2e-4
NUM_WORKERS=12

# Model parameters (must match your VAE and LDM)
# VAE from vae_run_3 was trained with latent_dim=32
LATENT_DIM=32
LATENT_SIZE=16
CONTOUR_CHANNELS=1

echo "Starting ControlNet training..."
echo "VAE Checkpoint: $VAE_CHECKPOINT"
echo "Data Path: $DATA_PATH"
echo "Save Dir: $SAVE_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Latent Dim: $LATENT_DIM"
echo "Latent Size: $LATENT_SIZE"
echo "Number of GPUs: $NUM_GPUS"

# Create output directory
mkdir -p $SAVE_DIR

# Run distributed training
torchrun --nproc_per_node=$NUM_GPUS \
    ../train_controlnet.py \
    --data_path $DATA_PATH \
    --csv_path $CSV_PATH \
    --save_dir $SAVE_DIR \
    --vae_checkpoint $VAE_CHECKPOINT \
    --unet_checkpoint $UNET_CHECKPOINT \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --latent_dim $LATENT_DIM \
    --latent_size $LATENT_SIZE \
    --contour_channels $CONTOUR_CHANNELS \
    --save_interval 5 \
    --val_interval 10 \
    --num_workers $NUM_WORKERS

echo "ControlNet training completed!"
echo "Checkpoints saved in: $SAVE_DIR" 