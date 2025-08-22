# FibLDM: Fine-Tuning Stable Diffusion for 2D Lung CT Scan Generation

<img width="1062" height="532" alt="image" src="https://github.com/user-attachments/assets/7825a26e-5d16-46f7-abd6-1700abd11023" />

---

**FibLDM** is a project focused on generating high-quality, 2D axial lung CT scans, particularly those exhibiting features of fibrosis. This repository is currently under active development.

### Project Overview
The primary goal of **FibLDM** is to create a robust generative model for medical imaging. We are leveraging a pre-trained **Stable Diffusion (v1.5)** model and adapting its U-Net component to our domain of medical imaging using **Low-Rank Adaptation (LoRA)**. This allows for efficient training on a custom dataset of latent representations of lung CT scans. The project utilizes modern deep learning libraries, including Hugging Face's `diffusers` and `peft`.

### Current Status
🚧 **This project is a work in progress.** 🚧

The current phase involves:
- Fine-tuning the model on a large, private dataset of lung CT latents.
- Experimenting with class conditioning (e.g., "healthy" vs. "fibrosis") and slice index conditioning to improve image quality and control.
- Integrating Weights & Biases (`wandb`) for experiment tracking and performance monitoring.

---

## Training Pipeline

The training process is managed via a shell script that controls all relevant hyperparameters.

### Step 1: Set up the Environment

First, create the conda environment from the provided `environment.yml` file.

```bash
conda env create -f environment.yml
conda activate fibldm
```

### Step 2: Configure the Training Run
Modify the hyperparameters at the top of the train_lora_unet.sh script to define your experiment.

```#!/bin/bash

# -- Experiment Configuration --
export DATA_DIR="/path/to/your/data"
export OUTPUT_DIR="./results/my_new_run"

# -- Hyperparameters --
LORA_RANK=32
LEARNING_RATE=5e-5
NUM_EPOCHS=150
# ... and so on
```

### Step 3: Start the Training
Execute the script to begin the fine-tuning process. The script supports multi-GPU training and will automatically log metrics and sample images to Weights & Biases if configured.
```
./scripts/train_lora_unet.sh
```
