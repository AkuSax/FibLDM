import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    latent_dir = os.path.join(args.latent_data_dir, 'latents')
    all_latents = []
    for fname in sorted(os.listdir(latent_dir)):
        if fname.endswith('.pt'):
            latent = torch.load(os.path.join(latent_dir, fname))
            all_latents.append(latent.flatten())
    all_latents = torch.cat(all_latents)
    mean = all_latents.mean().item()
    std = all_latents.std().item()
    print(f"Latent mean: {mean:.4f}")
    print(f"Latent std: {std:.4f}")
    plt.hist(all_latents.cpu().numpy(), bins=100, alpha=0.7)
    plt.title('Latent Value Distribution (Unnormalized)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(args.latent_data_dir, 'latent_distribution_hist.png'))
    print(f"Saved histogram to {os.path.join(args.latent_data_dir, 'latent_distribution_hist.png')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze VAE Latent Distribution')
    parser.add_argument('--latent_data_dir', type=str, required=True, help='Directory containing latents/')
    args = parser.parse_args()
    main(args) 