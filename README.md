# DDPM‑v2 – Multi‑GPU Diffusion for Contour Segmentation

---

**DDPM‑v2** is a research‑grade PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) tailored for pixel‑wise contour/segmentation tasks.  The codebase features:

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
├── requirements.txt      # Python deps
└── README.md             # <— you are here
```

---

## Quick start

### 1 · Install

```bash
conda create -n contourdiff python=3.10
conda activate contourdiff
pip install -r requirements.txt
```

### 2 · Prepare data

```
/hot/Yi-Kuan/Fibrosis/
 ├─ images/       *.png or *.tif
 └─ label.csv     # two‑column: id, mask‑path
```

Adapt `dataset.py` if your CSV layout differs.

### 3 · Single‑GPU smoke test

```bash
python main.py \
  --device cuda:0 \
  --batch_size 8 \
  --epochs 1
```

### 4 · Multi‑GPU training (2× A6000)

```bash
torchrun --nproc_per_node=2 \
         --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
         main.py \
           --batch_size 16 \
           --epochs 1000 \
           --data_dir /scratch/$USER/fibrosis_data \
           --save_dir  ./runs
```

*Effective* global batch = `nproc × --batch_size`.

Checkpoint files drop into `trained_models/` every 30 epochs; the best EMA model lands in `best_ddpm_model.pth`.

---

## Key files & APIs

| File                | What it does                                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **train\_utils.py** | `train()` loops over epochs with AMP, EMA, IoU early‑stop, and returns EMA samples + contours.  Modify `loss_fn` or add extra terms here.                        |
| **DDPM.py**         | `noise_image`, `sample_timesteps`, `sample()` implement the forward/reverse diffusion.  Cosine schedule is plugged in from `train_utils.cosine_beta_schedule()`. |
| **model.py**        | Plain UNet‑2D with timestep embedding.  Swap in Swin‑UNet, Mask2Former, etc., keeping `(x_in, t)` signature.                                                     |
| **dataset.py**      | Minimal `__getitem__` → `(image_tensor, contour_tensor)`.  Extend here for multi‑class masks.                                                                    |

---

## Road‑map / TODO

1. **Loss stack upgrade**
   - ✓ noise‑MSE  •  □ Dice  •  □ Boundary / Hausdorff  •  □ GAN adversarial
2. **Architecture variants**\
   □ Swin‑UNet  •  □ Mask2Former head  •  □ Diffusion transformer encoder
3. **Inference CLI / notebook**
4. **GitHub Actions** for linting + small data CI
5. **Weights & Biases logging** (optional)

---

## Citing

If you build on DDPM‑v2 for academic work, please cite:

```
@software{ddpm_v2_2025,
  author    = {Akul Saxena et al.},
  title     = {DDPM‑v2: Mixed‑precision, multi‑GPU diffusion for contour segmentation},
  year      = {2025},
  url       = {https://github.com/AkuSax/DDPM-v2}
}
```

---

## License

This project is released under the **MIT License** (see `LICENSE`).  Models trained on proprietary data remain the property of their respective owners.

