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
from diffusers import AutoencoderKL

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Enforce latent_dim=4 for SD VAE ---
    if args.use_sd_vae:
        if args.latent_dim != 4:
            print("[INFO] Overriding latent_dim to 4 for Stable Diffusion VAE.")
        args.latent_dim = 4

    # --- Load Models ---
    if args.use_sd_vae:
        print("Using Stable Diffusion VAE from diffusers...")
        model_id = "runwayml/stable-diffusion-v1-5"
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        # Do NOT load a custom checkpoint!
    else:
        print("Using custom VAE...")
        vae = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
        vae.load_state_dict(torch.load(args.vae_checkpoint))
    vae.eval()

    # --- Load Original Dataset ---
    print("Loading original dataset...")
    dataset = ContourDataset(label_file=args.csv_file, img_dir=args.img_dir, istransform=False)
    
    # --- Create Directories ---
    latent_dir = os.path.join(args.output_dir, "latents")
    contour_dir = os.path.join(args.output_dir, "contours")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(contour_dir, exist_ok=True)

    # --- PASS 1: Encode and Save Un-normalized Latents ---
    print("\n--- Pass 1: Encoding dataset and saving un-normalized latents... ---")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            image, contour = dataset[i]
            image = image.unsqueeze(0).to(device)
            # If using SD VAE, convert grayscale to 3 channels
            if args.use_sd_vae and image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            if args.use_sd_vae:
                latent_dist = vae.encode(image).latent_dist
                mu = latent_dist.mean
            else:
                mu, _ = vae.encode(image)
            # NO SCALING by 0.18215 anymore
            latent_path = os.path.join(latent_dir, f"{i}.pt")
            contour_path = os.path.join(contour_dir, f"{i}.pt")
            torch.save(mu.cpu(), latent_path)
            torch.save(contour.cpu(), contour_path)

    # --- CALCULATE GLOBAL STATS ---
    print("\n--- Calculating global mean and std of all latents... ---")
    all_latents = []
    for i in tqdm(range(len(dataset))):
        latent_path = os.path.join(latent_dir, f"{i}.pt")
        all_latents.append(torch.load(latent_path))
    all_latents_tensor = torch.cat(all_latents, dim=0)
    global_mean = all_latents_tensor.mean()
    global_std = all_latents_tensor.std()
    print(f"Global Mean: {global_mean.item():.6f}")
    print(f"Global Std:  {global_std.item():.6f}")
    # Save the stats for later use during inference
    stats_path = os.path.join(args.output_dir, "latent_stats.pt")
    torch.save({"mean": global_mean, "std": global_std}, stats_path)
    print(f"Saved latent stats to {stats_path}")

    # --- PASS 2: Normalize and Overwrite Latents ---
    print("\n--- Pass 2: Normalizing latents with global stats and overwriting... ---")
    manifest = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            latent_path = os.path.join(latent_dir, f"{i}.pt")
            latent = torch.load(latent_path)
            # Normalize the latent
            normalized_latent = (latent - global_mean) / global_std
            # Overwrite the file with the normalized latent
            torch.save(normalized_latent.cpu(), latent_path)
            original_img_path = dataset.img_labels.iloc[i, 0]
            manifest.append({"index": i, "original_file": original_img_path})
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join(args.output_dir, "manifest.csv"), index=False)
    print("\nNormalization complete. Manifest saved.")

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
    parser.add_argument("--use_sd_vae", action="store_true", help="Use Stable Diffusion VAE from diffusers instead of custom VAE.")
    
    args = parser.parse_args()
    main(args) 