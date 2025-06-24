import os
import torch
import argparse
import numpy as np


def main(args):
    latent_dirs = [os.path.join(args.latent_dataset_dir, 'latents'),
                   os.path.join(args.latent_dataset_dir, 'contours')]
    
    for latent_dir in latent_dirs:
        latent_files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.pt')])
        all_latents = []
        print(f"Found {len(latent_files)} latent files in {latent_dir}. Loading...")
        for fname in latent_files:
            latent = torch.load(os.path.join(latent_dir, fname))
            # Remove batch dim if present
            if latent.dim() == 4 and latent.shape[0] == 1:
                latent = latent.squeeze(0)
            all_latents.append(latent.cpu().numpy())
        
        all_latents = np.stack(all_latents, axis=0)  # shape: (N, C, H, W)
        print(f"All latents shape: {all_latents.shape}")
        mean = np.mean(all_latents)
        std = np.std(all_latents)
        print(f"Global mean: {mean:.4f}")
        print(f"Global std:  {std:.4f}")
        # Per-channel stats
        mean_c = np.mean(all_latents, axis=(0,2,3))
        std_c = np.std(all_latents, axis=(0,2,3))
        print("Per-channel mean:", np.round(mean_c, 4))
        print("Per-channel std:", np.round(std_c, 4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check VAE latent statistics.")
    parser.add_argument('--latent_dataset_dir', type=str, required=True, help='Path to the directory containing latents/ subdir.')
    args = parser.parse_args()
    main(args) 