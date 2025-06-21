#!/bin/bash

# Set environment variables for better stability
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=1800
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_SL=0
export NCCL_IB_TC=41
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1

# Clear any existing processes
echo "Stopping any existing training processes..."
pkill -f "main.py" || true
sleep 3

echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

echo "Starting stable training with reduced validation frequency..."
torchrun --nproc_per_node=2 main.py \
    --data_dir /hot/Yi-Kuan/Fibrosis/ \
    --csv_file /hot/Yi-Kuan/Fibrosis/label.csv \
    --arch unet2d \
    --batch_size 24 \
    --num_workers 4 \
    --epochs 500 \
    --use_amp \
    --use_compile \
    --metrics_interval 25 \
    --save_interval 25 \
    --losses mse,lpips \
    --lambda_mse 1.0 \
    --lambda_lpips 1.0 \
    --save_dir ./model_runs/full_run 