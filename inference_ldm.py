import torch
import argparse
import os
from torchvision.utils import save_image

from ddpm.diffusion import Diffusion
from unet2d import UNet2DLatent
from autoencoder import VAE

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Models ---
    # Load VAE
    vae = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
    vae.eval()
    print("Loaded VAE.")

    # Load LDM UNet
    unet = UNet2DLatent(
        img_size=args.latent_size,
        in_channels=args.latent_dim,
        out_channels=args.latent_dim
    ).to(device)
    unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device))
    unet.eval()
    print(f"Loaded LDM UNet from {args.unet_checkpoint}")

    # --- Setup Diffusion ---
    diffusion = Diffusion(
        noise_step=args.noise_steps,
        img_size=args.latent_size,
        device=str(device),
    )

    # --- Generate Samples (Unconditional) ---
    print(f"Generating {args.num_samples} unconditional samples...")
    with torch.no_grad():
        # Set diffusion to latent mode
        diffusion.is_latent = True
        # Generate random latents
        generated_latents = diffusion.sample(
            unet, n=args.num_samples, latent_dim=args.latent_dim
        )
        print("Latents generated. Decoding with VAE...")

        # Decode from latent space to image space
        # NOTE: The scaling factor is crucial for decoding
        generated_images = vae.decode(generated_latents / 0.18215)
    
    # --- Save Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "unconditional_samples.png")
    save_image(generated_images, output_path, normalize=True, nrow=4)
    
    print(f"Generation complete! Results saved in: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unconditional LDM Inference")
    
    parser.add_argument("--vae_checkpoint", type=str, required=True)
    parser.add_argument("--unet_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ldm_samples")
    
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--noise_steps", type=int, default=1000)
    
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=32)
    
    args = parser.parse_args()
    main(args)