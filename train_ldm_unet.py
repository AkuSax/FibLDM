import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR

# Local imports
from unet2d import UNet2DLatent
from dataset import LatentDataset
from ddpm.diffusion import Diffusion
from utils import EarlyStopper
from autoencoder import VAE
from train_utils import LatentFeatureExtractor

def main(args):
    torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load latent normalization stats ---
    if not os.path.exists(args.stats_path):
        raise FileNotFoundError(f"Latent stats file not found at {args.stats_path}. Please run encode_dataset.py again.")
    stats = torch.load(args.stats_path, map_location=device)
    latent_mean = stats["mean"]
    latent_std = stats["std"]
    print(f"Loaded latent stats: Mean={latent_mean.item():.4f}, Std={latent_std.item():.4f}")

    # --- Setup Debug Logger ---
    debug_log_path = os.path.join(args.save_dir, "debug_stats.log")
    debug_logger = logging.getLogger('debug_stats')
    debug_logger.setLevel(logging.INFO)
    debug_logger.propagate = False # Prevent printing to console/root logger
    file_handler = logging.FileHandler(debug_log_path, mode='w') # Overwrite log each run
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    debug_logger.addHandler(file_handler)
    print(f"Debug statistics will be saved to: {debug_log_path}")

    # --- Data ---
    dataset = LatentDataset(data_dir=args.latent_data_dir, latent_size=args.latent_size)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Training on {len(train_ds)} latent samples, validating on {len(val_ds)} samples.")

    # --- Models ---
    # LDM UNet Model
    model = UNet2DLatent(
        img_size=args.latent_size,
        in_channels=args.latent_dim,
        out_channels=args.latent_dim
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=20, mode='min')
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Load VAE for decoding samples (only needed for sampling)
    if not args.vae_checkpoint:
        print("Warning: No VAE checkpoint provided. Debug image sampling will be skipped.")
        vae = None
    else:
        print(f"Loading VAE from {args.vae_checkpoint} for image sampling...")
        vae = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
        vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
        vae.eval()
        print("VAE loaded.")

    # --- Diffusion ---
    diffusion = Diffusion(
        noise_step=args.noise_steps,
        img_size=args.latent_size,
        device=device,
        schedule_name=args.noise_schedule
    )

    # --- ADD: Initialize latent perceptual loss network ---
    latent_lpips = LatentFeatureExtractor(in_channels=args.latent_dim).to(device)
    latent_lpips.eval()

    # --- Training Loop ---
    metrics_csv = os.path.join(args.save_dir, 'ldm_unet_train_metrics.csv')
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, 'w') as f:
            f.write('epoch,avg_mse,avg_lpips,avg_total_loss\n')
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        total_mse = 0
        total_lpips = 0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for batch_idx, (latents, _) in enumerate(pbar):
            latents = latents.to(device)
            t = diffusion.sample_timesteps(latents.size(0)).to(device)
            x_t, noise = diffusion.noise_image(latents, t)
            optimizer.zero_grad()
            pred_noise = model(x_t, t)
            # --- Combined MSE + latent LPIPS loss ---
            loss_mse = F.mse_loss(pred_noise, noise)
            with torch.no_grad():
                feat_pred = latent_lpips(pred_noise)
                feat_target = latent_lpips(noise)
            loss_lpips = F.mse_loss(feat_pred, feat_target)
            loss = loss_mse + (args.lambda_lpips * loss_lpips)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_mse += loss_mse.item()
            total_lpips += loss_lpips.item()
            num_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{loss_mse.item():.4f}',
                'lpips': f'{loss_lpips.item():.4f}'
            })
        avg_train_loss = total_train_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_lpips = total_lpips / num_batches
        # --- Write metrics to CSV ---
        with open(metrics_csv, 'a') as f:
            f.write(f'{epoch+1},{avg_mse:.6f},{avg_lpips:.6f},{avg_train_loss:.6f}\n')

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]")
        with torch.no_grad():
            for batch_idx, (latents, _) in enumerate(val_loader):
                latents = latents.to(device)
                t = diffusion.sample_timesteps(latents.size(0)).to(device)
                x_t, noise = diffusion.noise_image(latents, t)
                pred_noise = model(x_t, t)
                loss = F.mse_loss(pred_noise, noise)
                total_val_loss += loss.item()
                pbar_val.set_postfix(loss=f"{loss.item():.4f}")
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        # --- Save Best Model Checkpoint ---
        if avg_val_loss < stopper.best_score:
            print(f"New best validation loss: {avg_val_loss:.4f}. Saving best model...")
            torch.save(model.state_dict(), os.path.join(args.save_dir, "unet_best.pth"))
        if stopper.early_stop(avg_val_loss):
            print("Early stopping triggered.")
            break

        # --- Interval Checkpointing and Debug Sampling ---
        if (epoch + 1) % args.save_interval == 0:
            # Save an interval checkpoint
            print(f"Saving interval checkpoint for epoch {epoch+1}...")
            interval_save_path = os.path.join(args.save_dir, f"unet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), interval_save_path)
            
            # Generate and save debug samples
            if vae is not None:
                print("Generating debug samples...")
                model.eval()
                with torch.no_grad():
                    diffusion.is_latent = True
                    generated_latents = diffusion.sample(
                        model, n=16, latent_dim=args.latent_dim, fast_sampling=True
                    )
                    # Un-normalize before decoding
                    unnormalized_latents = (generated_latents * latent_std) + latent_mean
                    generated_images = vae.decode(unnormalized_latents)
                
                # Save the grid of generated images
                sample_save_path = os.path.join(args.save_dir, f"sample_epoch_{epoch+1}.png")
                save_image(generated_images, sample_save_path, nrow=4, normalize=True)
                print(f"Saved debug samples to {sample_save_path}")

        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LDM UNet Training Script")
    # Paths and Directories
    parser.add_argument("--latent_data_dir", type=str, required=True, help="Directory with encoded latents and contours.")
    parser.add_argument("--save_dir", type=str, default="ldm_unet_checkpoint", help="Directory to save model.")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Path to the pre-trained VAE checkpoint for sampling.")
    parser.add_argument("--stats_path", type=str, required=True, help="Path to the latent_stats.pt file.")
    
    # Model Hyperparameters
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent channels (must match VAE and encoding).")
    parser.add_argument("--latent_size", type=int, default=16, help="Latent spatial size (must match VAE and encoding).")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--save_interval", type=int, default=10, help="Epoch interval to save model checkpoints and samples.")
    
    # Diffusion Hyperparameters
    parser.add_argument("--noise_steps", type=int, default=1000, help="Diffusion steps.")
    parser.add_argument("--noise_schedule", type=str, default='cosine', choices=['cosine', 'linear'], help="Noise schedule.")
    
    # --- ADD: lambda_lpips argument ---
    parser.add_argument("--lambda_lpips", type=float, default=0.5, help="Weight for the latent LPIPS loss term.")
    
    args = parser.parse_args()
    main(args) 