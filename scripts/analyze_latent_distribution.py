import os
import sys
# Make sure the parent directory is in the path to find other modules if necessary
if '..' not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm # Import tqdm for the progress bar

def main(args):
    latent_dir = os.path.join(args.latent_data_dir, 'latents')
    
    # Get the list of all .pt files first
    latent_files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.pt')])
    num_latents = len(latent_files)
    print(f"Found {num_latents} latent files.")

    # --- Optimization 1: Pre-allocate the array ---
    # Load the first latent to determine the shape for pre-allocation
    first_latent = torch.load(os.path.join(latent_dir, latent_files[0]))
    latent_dim = first_latent.flatten().shape[0]
    
    # Create an empty NumPy array to hold all the data
    # This is much more memory-efficient than a large Python list of tensors
    all_latents_np = np.empty((num_latents, latent_dim), dtype=np.float32)

    # --- Optimization 2: Add tqdm for progress monitoring ---
    # Loop through the files with a progress bar
    for i, fname in enumerate(tqdm(latent_files, desc="Loading and processing latents")):
        latent_path = os.path.join(latent_dir, fname)
        latent = torch.load(latent_path)
        # Fill the pre-allocated array row by row
        all_latents_np[i] = latent.flatten().cpu().numpy()

    print("Successfully loaded all latents.")

    # --- Analysis (no changes needed here) ---
    mean = all_latents_np.mean()
    std = all_latents_np.std()
    print(f"Latent mean: {mean:.4f}")
    print(f"Latent std: {std:.4f}")

    print("Generating histogram...")
    plt.hist(all_latents_np.flatten(), bins=100, alpha=0.7)
    plt.title('Latent Value Distribution (Unnormalized)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    save_path = os.path.join(args.latent_data_dir, 'latent_distribution_hist.png')
    plt.savefig(save_path)
    print(f"Saved histogram to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze VAE Latent Distribution')
    parser.add_argument('--latent_data_dir', type=str, required=True, help='Directory containing latents/')
    args = parser.parse_args()
    main(args)