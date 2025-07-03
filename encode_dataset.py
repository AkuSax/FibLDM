import torch
import os
import argparse
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from skimage.measure import find_contours
from skimage.draw import polygon
import matplotlib.pyplot as plt

from dataset import ContourDataset
from autoencoder import VAE

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Models ---
    print("Loading trained VAE model...")
    vae = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint))
    vae.eval()

    # --- Load Original Dataset ---
    print("Loading original dataset...")
    dataset = ContourDataset(label_file=args.csv_file, img_dir=args.img_dir, istransform=True)
    
    # --- Create Directories for Latent Data ---
    latent_dir = os.path.join(args.output_dir, "latents")
    contour_dir = os.path.join(args.output_dir, "contours")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(contour_dir, exist_ok=True)

    print(f"Encoding dataset and saving to {args.output_dir}...")
    with torch.no_grad():
        manifest = []
        for i in tqdm(range(len(dataset))):
            image, contour = dataset[i]
            image = image.unsqueeze(0).to(device)
            # Debug: Check image input to VAE encoder before encoding
            print(f"Encoding Loop Item {i}: Image input to VAE - Shape: {image.shape}, Min: {image.min():.4f}, Max: {image.max():.4f}")
            mu, _ = vae.encode(image)
            mu = mu * 0.18215
            # Debug: Check VAE mu output shape
            print(f"Encoding Loop Item {i}: VAE mu shape: {mu.shape}")
            assert mu.shape[1] == args.latent_dim, f"VAE mu channels ({mu.shape[1]}) do not match latent_dim ({args.latent_dim})."
            assert mu.shape[2] == 16 and mu.shape[3] == 16, f"VAE mu spatial size expected (16,16), got ({mu.shape[2]}, {mu.shape[3]})."
            # Debug: Check latent properties before saving
            print(f"Encoding Loop Item {i}: Latent (mu) - Shape: {mu.shape}, Min: {mu.min():.4f}, Max: {mu.max():.4f}")
            assert mu.ndim == 4 and mu.shape[0] == 1, "Latent should have batch dim of 1."
            latent_path = os.path.join(latent_dir, f"{i}.pt")
            contour_path = os.path.join(contour_dir, f"{i}.pt")
            torch.save(mu.cpu(), latent_path)
            torch.save(contour.cpu(), contour_path)
            original_img_path = dataset.img_labels.iloc[i, 0]
            manifest.append({"index": i, "original_file": original_img_path})
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join(args.output_dir, "manifest.csv"), index=False)
    print("Encoding complete. Manifest saved.")

    print("Debug image generation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Encoding Script")
    # Paths
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the image directory (e.g., ./data/).")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Path to the trained VAE model checkpoint (e.g., vae_run_1/vae_best.pth).")
    parser.add_argument("--output_dir", type=str, default="./data/latents_dataset", help="Directory to save the encoded latents and contours.")
    
    # Model Hyperparameters
    parser.add_argument("--latent_dim", type=int, default=8, help="Dimension of the VAE latent space. Must match the trained VAE.")
    
    args = parser.parse_args()
    main(args) 