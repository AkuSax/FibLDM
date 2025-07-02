import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F
from unet2d import UNet2DLatent
from dataset import LatentDataset
from ddpm.diffusion import Diffusion
from utils import EarlyStopper


def main(args):
    torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Data ---
    dataset = LatentDataset(data_dir=args.latent_data_dir, latent_size=args.latent_size)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Training on {len(train_ds)} latent samples, validating on {len(val_ds)} samples.")

    # --- Model ---
    model = UNet2DLatent(
        img_size=args.latent_size,
        in_channels=args.latent_dim,
        out_channels=args.latent_dim
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=10, mode='min')

    # --- Diffusion ---
    diffusion = Diffusion(
        noise_step=args.noise_steps,
        img_size=args.latent_size,
        device=device,
        schedule_name=args.noise_schedule
    )

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for batch_idx, (latents, _) in enumerate(train_loader):
            latents = latents.to(device)
            t = diffusion.sample_timesteps(latents.size(0)).to(device)
            x_t, noise = diffusion.noise_image(latents, t)
            optimizer.zero_grad()
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = total_train_loss / len(train_loader)

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

        # --- Save Checkpoint ---
        if avg_val_loss < stopper.best_score:
            print(f"New best validation loss: {avg_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(args.save_dir, "unet_best.pth"))
        if stopper.early_stop(avg_val_loss):
            print("Early stopping triggered.")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LDM UNet Training Script")
    parser.add_argument("--latent_data_dir", type=str, required=True, help="Directory with encoded latents and contours.")
    parser.add_argument("--save_dir", type=str, default="ldm_unet_checkpoint", help="Directory to save model.")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent channels (must match VAE and encoding).")
    parser.add_argument("--latent_size", type=int, default=16, help="Latent spatial size (must match VAE and encoding).")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--noise_steps", type=int, default=1000, help="Diffusion steps.")
    parser.add_argument("--noise_schedule", type=str, default='cosine', choices=['cosine', 'linear'], help="Noise schedule.")
    args = parser.parse_args()
    main(args) 