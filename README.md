# DDPM‑v2 – Multi‑GPU Diffusion for Contour Segmentation

---

**DDPM‑v2** is a PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) tailored for pixel‑wise contour/segmentation tasks.  The codebase features:

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
├── main.py               # DDP entry point
├── train_utils.py        # AMP + EMA + early‑stop loop
├── model.py              # UNet2D backbone
├── DDPM.py               # Diffusion utilities (noise, sampling)
├── dataset.py            # ContourDataset wrapper
├── utils.py              # save_images, helper I/O
├── trained_models/       # periodic ckpts (.pt)
└── README.md             # <— you are here
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