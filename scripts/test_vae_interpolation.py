import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse
from torchvision.utils import save_image
from autoencoder import VAE
from dataset import ContourDataset

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
    vae.eval()
    dataset = ContourDataset(label_file=args.csv_file, img_dir=args.img_dir, istransform=False)
    idx_a, idx_b = args.idx_a, args.idx_b
    img_a, _ = dataset[idx_a]
    img_b, _ = dataset[idx_b]
    img_a = img_a.unsqueeze(0).to(device)
    img_b = img_b.unsqueeze(0).to(device)
    with torch.no_grad():
        mu_a, _ = vae.encode(img_a)
        mu_b, _ = vae.encode(img_b)
        # Interpolate
        alphas = torch.linspace(0, 1, steps=args.num_interp).to(device)
        latents = torch.stack([(1 - a) * mu_a + a * mu_b for a in alphas], dim=0).squeeze(1)
        decoded = vae.decode(latents)
        # Save grid
        save_image(decoded.cpu(), os.path.join(args.output_dir, 'vae_interp_grid.png'), nrow=args.num_interp, normalize=True)
        print(f"Saved VAE interpolation to {os.path.join(args.output_dir, 'vae_interp_grid.png')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Latent Interpolation Visualization')
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./vae_interp_samples')
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--idx_a', type=int, default=0)
    parser.add_argument('--idx_b', type=int, default=1)
    parser.add_argument('--num_interp', type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args) 