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
./scripts/train_vae.sh
```
*This will create a `vae_run_1/` directory containing the `vae_best.pth` model checkpoint.*

### Step 2: Encode the Dataset into Latents

Once the VAE is trained, use it to encode the entire image dataset into latent vectors. This pre-computation step makes the main LDM training much faster.

```bash
# Run the encoding script, pointing it to your trained VAE.
python encode_dataset.py \
    --csv_file ./data/small_label.csv \
    --img_dir ./data/ \
    --vae_checkpoint ./vae_run_1/vae_best.pth \
    --output_dir ./data/latents_dataset
```