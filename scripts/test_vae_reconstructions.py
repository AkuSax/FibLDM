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
    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        images, _ = zip(*[dataset[i] for i in range(args.num_samples)])
        images = torch.stack(images).to(device)
        recon, _, _ = vae(images)
        # Save grid: originals (top), reconstructions (bottom)
        comparison = torch.cat([images[:args.num_show], recon[:args.num_show]])
        save_image(comparison.cpu(), os.path.join(args.output_dir, 'vae_recon_grid.png'), nrow=args.num_show, normalize=True)
        print(f"Saved VAE reconstructions to {os.path.join(args.output_dir, 'vae_recon_grid.png')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Reconstruction Visualization')
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./vae_recon_samples')
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--num_show', type=int, default=8)
    args = parser.parse_args()
    main(args) 