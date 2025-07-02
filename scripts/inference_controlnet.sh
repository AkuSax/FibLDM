#!/bin/bash

# ControlNet Inference Script
# This script generates images using a trained ControlNet

# Model paths
VAE_CHECKPOINT="../model_runs/vae_run_3/vae_best.pth"  # Updated to match training
CONTROLNET_CHECKPOINT="../model_runs/controlnet_run_5/controlnet_epoch_25.pth"  # Adjust epoch number

# Data paths
DATA_PATH="/hot/Yi-Kuan/Fibrosis"  # Updated to match training
CSV_PATH="/hot/Yi-Kuan/Fibrosis/label.csv"  # Updated to match training

# Output directory
OUTPUT_DIR="../model_runs/controlnet_run_5/controlnet_samples"

# Generation parameters
NUM_SAMPLES=20
NOISE_STEPS=1000
SAMPLING_STEPS=100

# Model parameters (must match training and VAE)
LATENT_DIM=32
LATENT_SIZE=16
CONTOUR_CHANNELS=1

echo "Starting ControlNet inference..."
echo "VAE Checkpoint: $VAE_CHECKPOINT"
echo "ControlNet Checkpoint: $CONTROLNET_CHECKPOINT"
echo "Output Dir: $OUTPUT_DIR"
echo "Number of samples: $NUM_SAMPLES"
echo "Latent Dim: $LATENT_DIM"
echo "Latent Size: $LATENT_SIZE"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run inference
python ../inference_controlnet.py \
    --vae_checkpoint $VAE_CHECKPOINT \
    --controlnet_checkpoint $CONTROLNET_CHECKPOINT \
    --data_path $DATA_PATH \
    --csv_path $CSV_PATH \
    --output_dir $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES \
    --noise_steps $NOISE_STEPS \
    --sampling_steps $SAMPLING_STEPS \
    --latent_dim $LATENT_DIM \
    --latent_size $LATENT_SIZE \
    --contour_channels $CONTOUR_CHANNELS

echo "Inference completed!"
echo "Results saved in: $OUTPUT_DIR" 