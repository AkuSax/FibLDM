#!/usr/bin/env python3

import torch
import os
import argparse
import numpy as np

def main(args):
    """Check the contour data in the latent dataset to verify preprocessing."""
    
    contour_dir = os.path.join(args.latent_datapath, "contours")
    latent_dir = os.path.join(args.latent_datapath, "latents")
    
    if not os.path.exists(contour_dir):
        print(f"Error: Contour directory not found at {contour_dir}")
        return
    
    if not os.path.exists(latent_dir):
        print(f"Error: Latent directory not found at {latent_dir}")
        return
    
    # Get list of contour files
    contour_files = sorted([f for f in os.listdir(contour_dir) if f.endswith('.pt')])
    print(f"Found {len(contour_files)} contour files")
    
    if len(contour_files) == 0:
        print("No contour files found!")
        return
    
    # Check first few contours
    print("\n" + "="*50)
    print("CONTOUR DATA INSPECTION")
    print("="*50)
    
    for i in range(min(5, len(contour_files))):
        contour_path = os.path.join(contour_dir, contour_files[i])
        contour = torch.load(contour_path)
        
        print(f"\nContour {i} ({contour_files[i]}):")
        print(f"  Shape: {contour.shape}")
        print(f"  Data type: {contour.dtype}")
        print(f"  Min value: {contour.min():.6f}")
        print(f"  Max value: {contour.max():.6f}")
        print(f"  Mean value: {contour.mean():.6f}")
        print(f"  Unique values: {torch.unique(contour)}")
        
        # Check if it's properly binarized
        unique_vals = torch.unique(contour)
        if len(unique_vals) == 2 and 0.0 in unique_vals and 1.0 in unique_vals:
            print(f"  ✓ Properly binarized (binary mask)")
        elif len(unique_vals) <= 10:
            print(f"  ⚠ Partially binarized (few unique values)")
        else:
            print(f"  ✗ Not binarized (many unique values)")
    
    # Check latent data too
    print("\n" + "="*50)
    print("LATENT DATA INSPECTION")
    print("="*50)
    
    latent_files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.pt')])
    if len(latent_files) > 0:
        latent_path = os.path.join(latent_dir, latent_files[0])
        latent = torch.load(latent_path)
        
        print(f"\nFirst latent ({latent_files[0]}):")
        print(f"  Shape: {latent.shape}")
        print(f"  Data type: {latent.dtype}")
        print(f"  Min value: {latent.min():.6f}")
        print(f"  Max value: {latent.max():.6f}")
        print(f"  Mean value: {latent.mean():.6f}")
        print(f"  Std value: {latent.std():.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check contour data in latent dataset")
    parser.add_argument('--latent_datapath', type=str, required=True, 
                       help='Path to the latent dataset directory (e.g., ../data32)')
    args = parser.parse_args()
    main(args) 