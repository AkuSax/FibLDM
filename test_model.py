#!/usr/bin/env python3

import torch
import torch.nn as nn
from unet2d import UNet2DLatent
import matplotlib.pyplot as plt
import numpy as np

def test_unet_latent():
    """Test the UNet2DLatent model to ensure it's working correctly."""
    
    # Model parameters (matching your training setup)
    img_size = 16
    in_channels = 33  # latent_dim (32) + contour_channels (1)
    out_channels = 32  # latent_dim
    
    print(f"Testing UNet2DLatent with:")
    print(f"  img_size: {img_size}")
    print(f"  in_channels: {in_channels}")
    print(f"  out_channels: {out_channels}")
    
    # Create model
    model = UNet2DLatent(img_size=img_size, in_channels=in_channels, out_channels=out_channels)
    model.eval()
    
    # Test inputs
    batch_size = 4
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    t1 = torch.randint(1, 1000, (batch_size,))  # Different timesteps
    t2 = torch.randint(1, 1000, (batch_size,))  # Different timesteps
    # Dummy contour input for FiLM
    contour = torch.randn(batch_size, 1, img_size, img_size)
    
    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  t1: {t1.shape}")
    print(f"  t2: {t2.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output1 = model(x, t1, contour)
        output2 = model(x, t2, contour)
        
        print(f"\nOutput shapes:")
        print(f"  output1: {output1.shape}")
        print(f"  output2: {output2.shape}")
        
        # Check that outputs are different for different timesteps
        # (this verifies time conditioning is working)
        diff = torch.abs(output1 - output2).mean()
        print(f"\nMean difference between outputs with different timesteps: {diff:.6f}")
        
        if diff > 1e-6:
            print("✓ Time conditioning is working (outputs differ for different timesteps)")
        else:
            print("✗ Time conditioning may not be working (outputs are identical)")
        
        # Check output statistics
        print(f"\nOutput statistics:")
        print(f"  output1 - mean: {output1.mean():.6f}, std: {output1.std():.6f}")
        print(f"  output2 - mean: {output2.mean():.6f}, std: {output2.std():.6f}")
        
        # Check for NaN or inf values
        if torch.isnan(output1).any() or torch.isinf(output1).any():
            print("✗ Output1 contains NaN or inf values")
        else:
            print("✓ Output1 is finite")
            
        if torch.isnan(output2).any() or torch.isinf(output2).any():
            print("✗ Output2 contains NaN or inf values")
        else:
            print("✓ Output2 is finite")

def test_time_embedding():
    """Test the time embedding specifically."""
    from unet2d import SinusoidalPositionEmbeddings
    
    time_dim = 128
    time_mlp = nn.Sequential(
        SinusoidalPositionEmbeddings(time_dim),
        nn.Linear(time_dim, time_dim * 4),
        nn.GELU(),
        nn.Linear(time_dim * 4, time_dim),
    )
    
    # Test with different timesteps
    t1 = torch.tensor([100])
    t2 = torch.tensor([500])
    
    with torch.no_grad():
        emb1 = time_mlp(t1)
        emb2 = time_mlp(t2)
        
        print(f"\nTime embedding test:")
        print(f"  t1={t1.item()}, embedding shape: {emb1.shape}")
        print(f"  t2={t2.item()}, embedding shape: {emb2.shape}")
        print(f"  Embeddings different: {torch.abs(emb1 - emb2).mean():.6f}")

if __name__ == "__main__":
    print("=== UNet2DLatent Model Test ===\n")
    test_time_embedding()
    test_unet_latent()
    print("\n=== Test Complete ===") 