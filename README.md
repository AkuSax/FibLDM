# FibLDM: Latent Diffusion Model for Fibrosis Synthesis

---

**FibLDM** is a high-performance, two-stage generative model in PyTorch for synthesizing medically-plausible CT scans of lungs with fibrosis. It is a **Latent Diffusion Model (LDM)**, which operates by first learning a compressed latent representation of the data and then running the diffusion process in that much smaller, more efficient space.

This approach significantly reduces computational requirements and training time compared to traditional pixel-space diffusion models.

### Core Architecture
1.  **Stage 1: Autoencoder (VAE)**
    -   A `Variational Autoencoder` is trained to compress high-resolution 256x256 images into a small `(8, 16, 16)` latent space.
    -   The VAE's decoder is used at the very end of the pipeline to transform generated latents back into full-resolution images.

2.  **Stage 2: Latent Diffusion Model**
    -   A conditional U-Net is trained entirely in the latent space.
    -   It learns to reverse a diffusion process, generating new latent vectors conditioned on a downsampled lung contour map.

### Key Features
-   **Efficient Latent Space Operation:** Drastically faster training and lower memory usage.
-   **High-Fidelity Generation:** The power of diffusion models combined with the perceptual quality of a well-trained VAE.
-   **Stable Training:** Implements a cosine noise schedule and modern training optimizations.
-   **Performance-Optimized:** Full support for DistributedDataParallel (DDP) and Automatic Mixed Precision (AMP).
-   **Integrated Validation:** On-the-fly decoding of generated samples for visual debugging and calculating realism metrics (FID, KID, LPIPS, SSIM) against real images.

---

## Training Pipeline

Training is a three-step process. You must complete them in order.

### Step 1: Train the Autoencoder

First, train the VAE on the full-resolution images. This script will save the best-performing model checkpoint based on validation loss.

```bash
# From the project root, run the VAE training script.
# Make sure to provide the correct paths to your data.
./train_vae.sh
```
*This will create a `vae_run_1/` directory containing the `vae_best.pth` model checkpoint.*

### Step 2: Encode the Dataset into Latents

Once the VAE is trained, use it to encode the entire image dataset into latent vectors. This pre-computation step makes the main LDM training much faster.

```bash
# Run the encoding script, pointing it to your trained VAE.
python encode_dataset.py \
    --csv_file ./small_label.csv \
    --img_dir /hot/Yi-Kuan/Fibrosis/ \
    --vae_checkpoint ./vae_run_1/vae_best.pth \
    --output_dir ./data/latents_dataset
```
*This will create a `./data/latents_dataset/` directory containing the latents, contours, and a manifest file.*

### Step 3: Train the Latent Diffusion Model

Finally, train the U-Net in the latent space. This script uses DDP for multi-GPU training and points to the latent dataset you just created.

```bash
# Use your existing training script, now configured for LDM.
# Ensure arguments in train_stable.sh point to the latent data.
./train_stable.sh
```
*This script should be configured to run `main.py` with `dataset_type=latent` and other relevant LDM arguments.*

---

## Directory Structure
```
FibLDM/
├── ddpm/
│   ├── diffusion.py        # Core DDPM forward/reverse process logic
│   └── losses.py           # Loss function registry
├── autoencoder.py          # The VAE model for the first stage
├── train_autoencoder.py    # Training script for the VAE
├── encode_dataset.py       # Script to pre-compute latents
├── unet2d.py               # U-Net architecture (operates on latents)
├── main.py                 # DDP entry point for LDM training
├── train_utils.py          # Core LDM training, validation, and sampling logic
├── dataset.py              # Contains both ContourDataset and LatentDataset
├── metrics.py              # Realism metrics (FID, KID, LPIPS, SSIM)
├── utils.py                # Helper classes (EarlyStopper, EMA)
└── requirements.txt        # Dependencies
```

