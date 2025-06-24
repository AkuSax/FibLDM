import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import argparse
import os
from tqdm import tqdm
import numpy as np

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
except ImportError:
    print("Please install torchmetrics: pip install torchmetrics")
    exit()

from dataset import ContourDataset
from autoencoder import VAE


def calculate_metrics_v2(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    print("Loading trained VAE model...")
    model = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
    try:
        model.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
    except Exception as e:
        print(f"Error loading state dictionary: {e}")
        return
    model.eval()
    print("Model loaded successfully.")

    # --- Load Dataset ---
    torch.manual_seed(42)
    print("Loading dataset...")
    full_dataset = ContourDataset(label_file=args.label_file, img_dir=args.data_dir)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_ds = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Evaluating on {len(val_ds)} validation samples.")

    # --- Initialize Metrics ---
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    total_mse_full, total_ssim_full = 0.0, 0.0
    total_mse_masked, total_ssim_masked = 0.0, 0.0

    pbar = tqdm(val_loader, desc="Calculating Metrics on Validation Set")
    with torch.no_grad():
        for i, (original_images, contours) in enumerate(pbar):
            original_images = original_images.to(device)
            contours = contours.to(device) # The contour is our mask

            # Generate reconstructions
            recon_images, _, _ = model(original_images)

            # --- Full Image Metrics ---
            total_mse_full += F.mse_loss(recon_images, original_images).item()
            original_norm_full = (original_images + 1) / 2
            recon_norm_full = (recon_images + 1) / 2
            total_ssim_full += ssim_metric(recon_norm_full, original_norm_full).item()

            # --- Masked Metrics ---
            # Apply the contour as a mask. Use clone() to avoid in-place modification errors.
            original_masked = original_images.clone() * contours
            recon_masked = recon_images.clone() * contours
            
            # Calculate MSE only on the masked region
            # We add a small epsilon to the sum of the mask to avoid division by zero if a contour is empty
            mse_masked = F.mse_loss(recon_masked, original_masked, reduction='sum') / (torch.sum(contours) + 1e-8)
            total_mse_masked += mse_masked.item()
            
            # Normalize masked images for SSIM
            original_norm_masked = (original_masked + 1) / 2
            recon_norm_masked = (recon_masked + 1) / 2
            total_ssim_masked += ssim_metric(recon_norm_masked, original_norm_masked).item()


    # --- Calculate and Print Averages ---
    num_batches = len(val_loader)
    avg_mse_full = total_mse_full / num_batches
    avg_ssim_full = total_ssim_full / num_batches
    avg_mse_masked = total_mse_masked / num_batches
    avg_ssim_masked = total_ssim_masked / num_batches

    print("\n--- Full Image Metrics (Potentially Misleading) ---")
    print(f"Average MSE:  {avg_mse_full:.6f}")
    print(f"Average SSIM: {avg_ssim_full:.6f}")
    print("--------------------------------------------------")
    print("\n--- MASKED Metrics (Region of Interest Only) ---")
    print(f"Average MSE on Mask:  {avg_mse_masked:.6f}")
    print(f"Average SSIM on Mask: {avg_ssim_masked:.6f}")
    print("------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VAE Reconstruction Metrics Calculation Script (V2 with Masking)")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Path to the trained VAE checkpoint.")
    parser.add_argument("--label_file", type=str, required=True, help="Path to the data CSV file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the image directory.")
    parser.add_argument("--latent_dim", type=int, required=True, help="Latent dimension of the VAE model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    args = parser.parse_args()
    calculate_metrics_v2(args)