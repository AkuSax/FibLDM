#!/usr/bin/env python3

# Simple script to verify VAE dimensions
# This doesn't require torch, just basic math

def calculate_latent_size(input_size, num_downsampling_layers):
    """Calculate latent size based on input size and number of downsampling layers."""
    size = input_size
    for _ in range(num_downsampling_layers):
        size = size // 2
    return size

def main():
    print("=== VAE Dimension Verification ===")
    
    # From train_vae.sh, we know:
    latent_dim = 32  # This was used in vae_run_3
    input_size = 256  # Standard input size
    num_downsampling = 4  # 4 Conv2d layers with stride=2
    
    latent_size = calculate_latent_size(input_size, num_downsampling)
    
    print(f"Input image size: {input_size}x{input_size}")
    print(f"Number of downsampling layers: {num_downsampling}")
    print(f"Calculated latent size: {latent_size}x{latent_size}")
    print(f"Latent dimension (channels): {latent_dim}")
    print(f"Total latent space: {latent_dim} channels × {latent_size} × {latent_size} = {latent_dim * latent_size * latent_size}")
    
    print("\n=== Recommended Settings ===")
    print("For train_controlnet.sh, use:")
    print(f"LATENT_DIM={latent_dim}")
    print(f"LATENT_SIZE={latent_size}")
    
    print("\n=== Current vs Correct Settings ===")
    print("Current (incorrect):")
    print("  LATENT_DIM=8")
    print("  LATENT_SIZE=16")
    print("\nCorrect:")
    print(f"  LATENT_DIM={latent_dim}")
    print(f"  LATENT_SIZE={latent_size}")

if __name__ == "__main__":
    main() 