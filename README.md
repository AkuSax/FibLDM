# DDPM-v2: Conditional Contour-Guided Diffusion for Fibrosis CT Generation

---

**DDPM‑v2** is a PyTorch implementation of a conditional Denoising Diffusion Probabilistic Model (DDPM), tuned for generating CT scans of lungs with fibrosis based on a conditioning input of a lung contour.

The codebase is built for performance and scalability, featuring:

-   **Conditional Image Generation:** Generates high-fidelity images conditioned on an input tensor.
-   **Cosine β‑schedule:** Implements the improved noise schedule from Nichol & Dhariwal, 2021 for stable training.
-   **Exponential Moving Average (EMA):** Uses an EMA of model weights for producing robust and high-quality samples during inference.
-   **Perceptual-driven Early Stopping:** Uses LPIPS score to monitor validation performance and stop training when perceptual metrics no longer improve.
-   **Performance Optimizations:**
    -   Automatic Mixed-Precision (AMP) via `torch.amp` for faster training.
    -   Out-of-the-box DistributedDataParallel (DDP) support for multi-GPU training.
    -   Optimized data loading with pinned memory, prefetching, and persistent workers.
-   **Flexible Loss Functions:** Includes a registry for easily combining multiple loss terms, including MSE, LPIPS, and Adversarial loss.

---

## How to Train

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Training:**
    Use the provided script to start a DDP training session. The following command from `train_stable.sh` starts training on 2 GPUs:

    ```bash
    torchrun --nproc_per_node=2 main.py \
        --data_dir /path/to/your/data \
        --csv_file /path/to/your/labels.csv \
        --batch_size 24 \
        --epochs 500 \
        --use_amp \
        --use_compile \
        --losses mse,lpips \
        --lambda_mse 1.0 \
        --lambda_lpips 1.0
    ```

---

## Directory Structure

```
DDPM-v2/
├── ddpm/
│   ├── init.py
│   ├── diffusion.py            # Core DDPM forward/reverse process logic
│   └── losses.py               # Registry of loss functions (MSE, LPIPS, Adv, etc.)
├── models/
│   ├── init.py             # Model registry (get_model)
│   └── unet2d.py               # Primary UNet architecture for the generator
├── main.py                     # DDP entry point for training
├── train_utils.py              # Core training, validation, and EMA logic
├── dataset.py                  # Dataset class for loading image pairs
├── discriminator.py            # PatchGAN discriminator for adversarial loss
├── metrics.py                  # Realism metrics (FID, KID, LPIPS, SSIM)
├── utils.py                    # Helper classes like EarlyStopper and EMA
└── requirements.txt            # Project dependencies
```
---

## Key Files

| File | What it does |
| :--- | :--- |
| **train\_utils.py** | Contains the main `train()` function which loops over epochs, handles AMP, updates the EMA model, and runs validation. This is where loss functions are combined and applied. |
| **ddpm/diffusion.py** | Implements the `Diffusion` class, which handles the core DDPM logic: `noise_image` (forward process) and `sample` (reverse process/inference). |
| **models/unet2d.py** | A standard 2D U-Net with timestep embeddings, which serves as the core noise prediction model. |
| **main.py** | Parses command-line arguments and sets up the DDP environment, then calls the `train()` function from `train_utils.py`. |

