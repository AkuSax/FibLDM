import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse
from autoencoder import VAE
from dataset import ContourDataset
from metrics import RealismMetrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
    vae.eval()
    dataset = ContourDataset(label_file=args.csv_file, img_dir=args.img_dir, istransform=False)
    metrics = RealismMetrics(device=str(device), sync_on_compute=True)
    all_ssim = []
    all_lpips = []
    with torch.no_grad():
        for i in range(min(args.num_samples, len(dataset))):
            img, _ = dataset[i]
            img = img.unsqueeze(0).to(device)
            recon, _, _ = vae(img)
            recon_clamped = recon.clamp(-1, 1)
            img_clamped = img.clamp(-1, 1)
            ssim = metrics.ssim(recon_clamped, img_clamped).item()
            lpips = metrics.lpips(recon_clamped.repeat(1, 3, 1, 1), img_clamped.repeat(1, 3, 1, 1)).item()
            all_ssim.append(ssim)
            all_lpips.append(lpips)
    print(f"SSIM: mean={torch.tensor(all_ssim).mean():.4f}, std={torch.tensor(all_ssim).std():.4f}")
    print(f"LPIPS: mean={torch.tensor(all_lpips).mean():.4f}, std={torch.tensor(all_lpips).std():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Reconstruction Metrics')
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()
    main(args) 