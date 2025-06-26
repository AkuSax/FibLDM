import torch
import argparse
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np

from ddpm.diffusion import Diffusion
from unet2d import UNet2DLatent
from autoencoder import VAE
from controlnet import ControlNet, ControlNetUNet
from dataset import ControlNetDataset
from utils import readmat


def load_models(args, device):
    """Load pre-trained VAE, UNet, and ControlNet."""
    
    # Load VAE
    vae = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
    vae.eval()
    
    # Load UNet
    unet = UNet2DLatent(
        img_size=args.latent_size,
        in_channels=args.latent_dim + args.contour_channels,
        out_channels=args.latent_dim
    ).to(device)
    
    if args.unet_checkpoint:
        unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device))
    
    # Load ControlNet weights from checkpoint dict
    controlnet = ControlNet(unet=unet, conditioning_channels=1).to(device)
    ckpt = torch.load(args.controlnet_checkpoint, map_location=device)
    controlnet.load_state_dict(ckpt["controlnet_state_dict"])
    controlnet.eval()
    
    # Create combined model
    controlnet_unet = ControlNetUNet(unet=unet, controlnet=controlnet).to(device)
    controlnet_unet.eval()
    
    return vae, controlnet_unet


def generate_with_controlnet(args):
    """Generate images using ControlNet with contour conditioning."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    vae, controlnet_unet = load_models(args, device)
    
    # Setup diffusion
    diffusion = Diffusion(
        noise_step=args.noise_steps,
        img_size=args.latent_size,
        device=str(device),
        schedule_name=args.noise_schedule
    )
    
    # Load test dataset for conditioning
    dataset = ControlNetDataset(
        label_file=os.path.join(args.data_path, args.csv_path),
        img_dir=args.data_path,
        img_size=args.img_size,
        istransform=False  # No augmentation for inference
    )
    
    print(f"Generating {args.num_samples} samples...")
    
    with torch.no_grad():
        for i in range(args.num_samples):
            # Get a random sample from dataset for conditioning
            sample_idx = i % len(dataset)
            sample = dataset[sample_idx]
            conditioning_image = sample['conditioning_image'].unsqueeze(0).to(device)
            
            # Start from random noise in latent space
            x = torch.randn(1, args.latent_dim, args.latent_size, args.latent_size).to(device)
            
            # Denoising loop
            for t in reversed(range(0, args.noise_steps, args.sampling_steps)):
                t_batch = torch.ones(1).to(device) * t
                
                # Predict noise using ControlNet
                predicted_noise = controlnet_unet(x, t_batch, conditioning_image)
                
                # Denoise step
                alpha = diffusion.alpha[t]
                alpha_hat = diffusion.alpha_hat[t]
                beta = diffusion.beta[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
            # Decode from latent space to image space
            generated_image = vae.decode(x)
            
            # Save results
            output_path = os.path.join(args.output_dir, f"generated_{i:03d}.png")
            save_image(generated_image, output_path, normalize=True)
            
            # Save conditioning contour for reference
            contour_path = os.path.join(args.output_dir, f"contour_{i:03d}.png")
            save_image(conditioning_image, contour_path, normalize=True)
            
            print(f"Generated sample {i+1}/{args.num_samples}: {output_path}")
    
    print(f"Generation complete! Results saved in: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ControlNet Inference")
    
    # Model paths
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--unet_checkpoint", type=str, default=None, help="Path to UNet checkpoint (optional)")
    parser.add_argument("--controlnet_checkpoint", type=str, required=True, help="Path to ControlNet checkpoint")
    
    # Data paths
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--csv_path", type=str, default="label.csv", help="Path to CSV file")
    parser.add_argument("--output_dir", type=str, default="./controlnet_samples", help="Output directory")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--noise_steps", type=int, default=1000, help="Number of noise steps")
    parser.add_argument("--sampling_steps", type=int, default=10, help="Sampling step size")
    parser.add_argument("--noise_schedule", type=str, default='cosine', choices=['cosine', 'linear'])
    
    # Model parameters
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--latent_size", type=int, default=16, help="Latent size")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--contour_channels", type=int, default=1, help="Contour channels")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    generate_with_controlnet(args)


if __name__ == '__main__':
    main() 