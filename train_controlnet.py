import os
import argparse
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from ddpm.diffusion import Diffusion
from unet2d import UNet2DLatent, get_model
from dataset import ControlNetDataset
from autoencoder import VAE
from controlnet import ControlNet, ControlNetUNet
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast  # Add mixed precision support
from tqdm import tqdm  # Add tqdm for progress bars

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def setup_logging(save_dir):
    """Setup logging configuration with rotation to prevent massive log files."""
    os.makedirs(save_dir, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        os.path.join(save_dir, 'controlnet_training.log'),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    
    console_handler = logging.StreamHandler()
    
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def train_controlnet_proc(args):
    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    os.environ['NCCL_TIMEOUT'] = '1800'
    
    try:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        dist.init_process_group(backend="nccl")
        
        if local_rank == 0:
            logging.info(f"[Rank {local_rank}] Starting ControlNet training with {world_size} GPUs")
    except Exception as e:
        logging.error(f"[Rank {local_rank}] Failed to initialize distributed training: {e}")
        raise

    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        if local_rank == 0:
            logging.info(f"[Rank {local_rank}] Loading pre-trained models...")
        
        # Load pre-trained VAE
        vae = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
        vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
        vae.eval()
        vae.requires_grad_(False)  # Freeze VAE
        
        # Load pre-trained UNet
        unet = UNet2DLatent(
            img_size=args.latent_size,
            in_channels=args.latent_dim,
            out_channels=args.latent_dim
        ).to(device)
        
        if not args.unet_checkpoint:
            raise ValueError("You must provide --unet_checkpoint (a trained LDM UNet checkpoint) for ControlNet training.")
        unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device))
        logging.info(f"[Rank {local_rank}] Loaded pre-trained UNet from {args.unet_checkpoint}")
        # Freeze the main UNet
        unet.requires_grad_(False)
        
        # Create ControlNet
        controlnet = ControlNet(unet=unet, conditioning_channels=1).to(device)
        
        # Create combined model
        controlnet_unet = ControlNetUNet(unet=unet, controlnet=controlnet).to(device)
        controlnet_unet = DDP(controlnet_unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
        # Enable torch.compile for faster training (PyTorch 2.0+)
        try:
            controlnet_unet = torch.compile(controlnet_unet, mode="reduce-overhead")
            if local_rank == 0:
                logging.info(f"[Rank {local_rank}] Enabled torch.compile optimization")
        except Exception as e:
            if local_rank == 0:
                logging.warning(f"[Rank {local_rank}] torch.compile not available: {e}")
        
        if local_rank == 0:
            logging.info(f"[Rank {local_rank}] Loading dataset...")
        
        # Load dataset
        dataset = ControlNetDataset(
            label_file=os.path.join(args.data_path, args.csv_path),
            img_dir=args.data_path,
            img_size=args.img_size,
            latent_size=args.latent_size
        )
        
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )

        sampler = distributed.DistributedSampler(train_ds, seed=42)
        val_sampler = distributed.DistributedSampler(val_ds, shuffle=False, seed=42)

        num_workers = min(args.num_workers, 4)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            drop_last=False
        )

        # Optimizer only for ControlNet parameters
        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
        
        # Mixed precision training
        scaler = GradScaler()

        # Diffusion setup
        diffusion = Diffusion(
            noise_step=args.noise_steps,
            img_size=args.latent_size,
            device=str(device),
            schedule_name=args.noise_schedule
        )

        if local_rank == 0:
            logging.info(f"[Rank {local_rank}] Starting ControlNet training...")
        
        # Training loop
        for epoch in range(args.num_epochs):
            controlnet_unet.train()
            sampler.set_epoch(epoch)
            
            total_loss = 0.0
            num_batches = 0
            
            # Create progress bar for training
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [TRAIN]", 
                             leave=False, ncols=100) if local_rank == 0 else train_loader
            
            for batch_idx, batch in enumerate(train_pbar):
                images = batch['image'].to(device)
                conditioning_images = batch['conditioning_image'].to(device)
                
                # Encode images to latent space using frozen VAE
                with torch.no_grad():
                    mu, _ = vae.encode(images)
                    mu = mu * 0.18215  # Scale the latents as required by LDM
                
                # Sample timesteps
                t = diffusion.sample_timesteps(images.shape[0])
                
                # Add noise to latents
                noise = torch.randn_like(mu)
                noisy_latents = diffusion.noise_image(mu, t)[0]
                
                # Predict noise using ControlNet with mixed precision
                with autocast():
                    predicted_noise = controlnet_unet(noisy_latents, t, conditioning_images)
                    # Calculate loss
                    loss = F.mse_loss(predicted_noise, noise)
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar with current loss (only if it's a tqdm object)
                if local_rank == 0 and isinstance(train_pbar, tqdm):
                    avg_loss = total_loss / num_batches
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg': f'{avg_loss:.4f}',
                        'LR': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
            
            scheduler.step()
            
            # Validation
            if (epoch + 1) % args.val_interval == 0:
                controlnet_unet.eval()
                val_loss = 0.0
                val_batches = 0
                
                # Create progress bar for validation
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [VAL]", 
                               leave=False, ncols=100) if local_rank == 0 else val_loader
                
                with torch.no_grad():
                    for batch in val_pbar:
                        images = batch['image'].to(device)
                        conditioning_images = batch['conditioning_image'].to(device)
                        
                        mu, _ = vae.encode(images)
                        mu = mu * 0.18215
                        t = diffusion.sample_timesteps(images.shape[0])
                        noise = torch.randn_like(mu)
                        noisy_latents = diffusion.noise_image(mu, t)[0]
                        
                        predicted_noise = controlnet_unet(noisy_latents, t, conditioning_images)
                        batch_val_loss = F.mse_loss(predicted_noise, noise).item()
                        val_loss += batch_val_loss
                        val_batches += 1
                        
                        # Update validation progress bar
                        if local_rank == 0 and isinstance(val_pbar, tqdm):
                            avg_val_loss = val_loss / val_batches
                            val_pbar.set_postfix({'Val Loss': f'{avg_val_loss:.4f}'})
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
                
                if local_rank == 0:
                    logging.info(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {total_loss/num_batches:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0 and local_rank == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'controlnet_state_dict': controlnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': total_loss / num_batches,
                }
                
                checkpoint_path = os.path.join(args.save_dir, f'controlnet_epoch_{epoch+1}.pth')
                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        if local_rank == 0:
            logging.info("ControlNet training completed!")
            
    except Exception as e:
        logging.error(f"[Rank {local_rank}] Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="ControlNet Training for Latent Diffusion Model")
    
    # --- Paths and Data ---
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the image dataset directory.")
    parser.add_argument("--csv_path", type=str, default="label.csv", help="Path to the dataset CSV file, relative to data_path.")
    parser.add_argument("--save_dir", type=str, default="model_runs/controlnet_run_1", help="Path to save the ControlNet model and logs.")
    
    # --- Pre-trained Model Paths ---
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Path to the pre-trained VAE checkpoint.")
    parser.add_argument("--unet_checkpoint", type=str, required=True, help="Path to the pre-trained UNet checkpoint (REQUIRED: must be a trained LDM UNet, not randomly initialized).")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--num_epochs", type=int, default=100, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for ControlNet optimizer.")
    parser.add_argument("--save_interval", type=int, default=10, help="Epoch interval to save model checkpoints.")
    parser.add_argument("--val_interval", type=int, default=5, help="Epoch interval to run validation.")
    
    # --- Model Hyperparameters ---
    parser.add_argument("--img_size", type=int, default=256, help="Size of the original images.")
    parser.add_argument("--latent_size", type=int, default=16, help="Spatial size of the latent space.")
    parser.add_argument("--latent_dim", type=int, default=8, help="Number of channels in the latent space.")
    parser.add_argument("--contour_channels", type=int, default=1, help="Number of channels for the contour condition.")
    
    # --- Diffusion Hyperparameters ---
    parser.add_argument("--noise_steps", type=int, default=1000, help="Total number of steps in the diffusion process.")
    parser.add_argument("--noise_schedule", type=str, default='cosine', choices=['cosine', 'linear'], help="Noise schedule for the diffusion process.")

    filtered_args = [arg for arg in sys.argv[1:] if arg.strip()]
    args = parser.parse_args(filtered_args)

    setup_logging(args.save_dir)
    train_controlnet_proc(args)


if __name__ == '__main__':
    main() 