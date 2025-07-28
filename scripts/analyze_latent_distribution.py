# scripts/analyze_latent_distribution.py
import os
import h5py
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main(args):
    h5_path = os.path.join(args.latent_data_dir, "latents_dataset_subset.h5")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 dataset not found at {h5_path}. Please run the preprocessing script first.")

    print(f"Loading latents from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # Load the entire dataset into memory for analysis
        all_latents_np = f['latents'][:]

    print("Successfully loaded all latents.")

    # --- Analysis & Stat File Creation ---
    mean = torch.from_numpy(all_latents_np).mean()
    std = torch.from_numpy(all_latents_np).std()
    
    stats_path = os.path.join(args.latent_data_dir, "latent_stats.pt")
    torch.save({"mean": mean, "std": std}, stats_path)
    
    print(f"\nLatent mean: {mean.item():.4f}")
    print(f"Latent std:  {std.item():.4f}")
    print(f"SUCCESS: Saved new statistics file to {stats_path}")

    # --- Visualization ---
    print("\nGenerating histogram...")
    plt.hist(all_latents_np.flatten(), bins=100, alpha=0.7)
    plt.title('Latent Value Distribution (Unnormalized Subset)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    save_path = os.path.join(args.latent_data_dir, 'latent_distribution_hist_subset.png')
    plt.savefig(save_path)
    print(f"Saved histogram to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze VAE Latent Distribution from HDF5')
    parser.add_argument('--latent_data_dir', type=str, required=True, help='Directory containing the HDF5 file and where to save the stats.')
    args = parser.parse_args()
    main(args)