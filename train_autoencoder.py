import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torchvision.utils import save_image
import argparse
import os
from tqdm import tqdm
import time

# Local imports
from dataset import ContourDataset
from autoencoder import VAE
from utils import EarlyStopper
from torch.cuda.amp import GradScaler, autocast # ✨ Import AMP modules
from diffusers import AutoencoderKL

def vae_loss_function(recon_x, x, mu, logvar, kld_weight):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = (recon_loss + kld_weight * kld_loss) / x.size(0)
    return loss, recon_loss / x.size(0), kld_loss / x.size(0)

def main(args):
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    # --- Data ---
    image_size = 512 if args.use_sd_vae else 256
    full_dataset = ContourDataset(label_file=args.label_file, img_dir=args.data_dir, istransform=True, image_size=image_size)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples.")

    # --- Model and Optimizer ---
    if args.use_sd_vae:
        print("Using Stable Diffusion VAE from diffusers...")
        model_id = "runwayml/stable-diffusion-v1-5"
        model = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        model.train()  # Ensure model is in training mode
    else:
        print("Using custom VAE...")
        model = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
        model.train()  # Ensure model is in training mode
    # Compile the model for speedup (PyTorch 2.0+)
    model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=10, mode='min')
    
    scaler = GradScaler()

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for batch_idx, (images, _) in enumerate(pbar):
            batch_start = time.time()
            images = images.to(device)
            data_time = time.time() - batch_start
            optimizer.zero_grad()

            # --- SD VAE expects 3-channel RGB in [0,1] ---
            if args.use_sd_vae:
                if images.shape[1] == 1:
                    images_in = images.repeat(1, 3, 1, 1)  # (B, 3, H, W)
                else:
                    images_in = images
                images_in = (images_in + 1.0) / 2.0  # [-1,1] -> [0,1]
            else:
                images_in = images

            images_in = images_in.float()
            images_for_loss = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False) if args.use_sd_vae else images
            with autocast():
                if args.use_sd_vae:
                    encoder_out = model.encode(images_in)
                    z = encoder_out.latent_dist.sample()
                    decoder_out = model.decode(z)
                    recon_images = decoder_out.sample
                    mu = encoder_out.latent_dist.mean
                    logvar = encoder_out.latent_dist.logvar
                    recon_images = recon_images * 2.0 - 1.0  # [0,1] -> [-1,1]
                    # --- Fix: Downsample and grayscale recon_images to match input ---
                    recon_images = F.interpolate(recon_images, size=(256, 256), mode='bilinear', align_corners=False)
                    if recon_images.shape[1] == 3:
                        recon_images = 0.2989 * recon_images[:, 0:1] + 0.5870 * recon_images[:, 1:2] + 0.1140 * recon_images[:, 2:3]
                else:
                    recon_images, mu, logvar = model(images_in)
                loss, recon_loss, kld_loss = vae_loss_function(recon_images, images_for_loss, mu, logvar, args.kld_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step_time = time.time() - batch_start
            total_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon_loss.item():.4f}", kld=f"{kld_loss.item():.4f}")
            # --- Log timing and GPU memory every 100 batches ---
            if batch_idx % 100 == 0:
                try:
                    mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                except Exception:
                    mem_alloc = mem_reserved = -1
                print(f"[Batch {batch_idx}] Data loading: {data_time:.3f}s, Step: {step_time:.3f}s, GPU mem: {mem_alloc:.2f}GB alloc, {mem_reserved:.2f}GB reserved")
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]")
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(val_loader):
                images = images.to(device)
                if args.use_sd_vae:
                    if images.shape[1] == 1:
                        images_in = images.repeat(1, 3, 1, 1)
                    else:
                        images_in = images
                    images_in = (images_in + 1.0) / 2.0
                else:
                    images_in = images
                with autocast():
                    if args.use_sd_vae:
                        out = model(images_in)
                        recon_images = out.sample
                        mu = out.latent_dist.mean
                        logvar = out.latent_dist.logvar
                        recon_images = recon_images * 2.0 - 1.0
                        # --- Fix: Downsample and grayscale recon_images to match input ---
                        recon_images = F.interpolate(recon_images, size=(256, 256), mode='bilinear', align_corners=False)
                        if recon_images.shape[1] == 3:
                            recon_images = 0.2989 * recon_images[:, 0:1] + 0.5870 * recon_images[:, 1:2] + 0.1140 * recon_images[:, 2:3]
                    else:
                        recon_images, mu, logvar = model(images_in)
                    loss, _, _ = vae_loss_function(recon_images, images, mu, logvar, args.kld_weight)
                total_val_loss += loss.item()
                pbar_val.set_postfix(loss=f"{loss.item():.4f}")
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        # --- Save Checkpoint and Samples ---
        if avg_val_loss < stopper.best_score:
            print(f"New best validation loss: {avg_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(args.save_dir, "vae_best.pth"))
        if stopper.early_stop(avg_val_loss):
            print("Early stopping triggered.")
            break
        if (epoch + 1) % args.save_interval == 0:
            with torch.no_grad():
                val_images, _ = next(iter(val_loader))
                val_images = val_images.to(device)
                if args.use_sd_vae:
                    if val_images.shape[1] == 1:
                        val_images_in = val_images.repeat(1, 3, 1, 1)
                    else:
                        val_images_in = val_images
                    val_images_in = (val_images_in + 1.0) / 2.0
                    out = model(val_images_in)
                    recon_val = out.sample
                    recon_val = recon_val * 2.0 - 1.0
                else:
                    recon_val, _, _ = model(val_images)
                comparison = torch.cat([val_images[:8], recon_val[:8]])
                save_image(comparison.cpu(), os.path.join(args.save_dir, f"recon_{epoch+1}.png"), nrow=8, normalize=True)
                print(f"Saved reconstruction sample to {os.path.join(args.save_dir, f'recon_{epoch+1}.png')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VAE Training Script")
    parser.add_argument("--label_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="vae_checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--kld_weight", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--use_sd_vae", action="store_true", help="Use Stable Diffusion VAE from diffusers instead of custom VAE.")
    
    args = parser.parse_args()
    main(args)
