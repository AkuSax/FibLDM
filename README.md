# DDPM‑v2 – Diffusion for Contour Segmentation

---

**DDPM‑v2** is a PyTorch implementation of a conditional Denoising Diffusion Probabilistic Model (DDPM) tailored for pixel‑wise contour/segmentation tasks.  The codebase features:

- **Cosine β‑schedule** (Nichol & Dhariwal, 2021) for smoother noise variance.
- **Exponential Moving Average (EMA)** of model weights for robust sampling.
- **IoU‑driven early stopping** & periodic MSE/IoU checkpoints.
- Automatic **mixed‑precision (AMP)** training with the new `torch.amp` API.
- Out‑of‑the‑box **DistributedDataParallel (DDP)** support – scales to 2× RTX A6000 (or more).
- Data‑loader optimizations: pinned memory, prefetch, persistent workers.
- Simple hooks to add Dice / Boundary / Hausdorff / adversarial loss terms (see *Road‑map*).

> **Status:** experimental – tuned for a fibrosis‐mask dataset but easily adaptable

---

## Directory structure

```
DDPM-v2/
├── ddpm/
│   ├── __init__.py
│   ├── diffusion.py            # Core DDPM forward/reverse process
│   └── losses.py               # Registry of loss functions
├── models/
│   ├── __init__.py             # Model registry (get_model)
│   ├── unet2d.py               # Primary UNet architecture
│   ├── swin_unet.py            # Wrapper for Swin-UNet
│   └── mask2former.py          # Wrapper for Mask2Former
├── main.py                     # DDP entry point for training
├── train_utils.py              # Core training, validation, and EMA logic
├── dataset.py                  # ContourDataset for loading .mat files
├── discriminator.py            # PatchGAN discriminator for adversarial loss
├── metrics.py                  # FID, KID, LPIPS, SSIM realism metrics
├── utils.py                    # Helper I/O for .mat files, EarlyStopper
├── compare_models.ipynb        # Notebook for comparing model checkpoints
├── requirements.txt            # Project dependencies
```

---

## Key files

| File                | What it does                                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **train\_utils.py** | `train()` loops over epochs with AMP, EMA, IoU early‑stop, and returns EMA samples + contours.  Modify `loss_fn` or add extra terms here.                        |
| **DDPM.py**         | `noise_image`, `sample_timesteps`, `sample()` implement the forward/reverse diffusion.  Cosine schedule is plugged in from `train_utils.cosine_beta_schedule()`. |
| **model.py**        | Plain UNet‑2D with timestep embedding.  Swap in Swin‑UNet, Mask2Former, etc., keeping `(x_in, t)` signature.                                                     |
| **dataset.py**      | Minimal `__getitem__` → `(image_tensor, contour_tensor)`.  Extend here for multi‑class masks.                                                                    |

---
