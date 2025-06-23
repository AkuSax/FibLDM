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
from unet2d import UNet2D, get_model
from dataset import ContourDataset, LatentDataset
from discriminator import PatchGANDiscriminator
from train_utils import train as ddpm_train, cosine_beta_schedule
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def setup_logging(save_dir):
    """Setup logging configuration with rotation to prevent massive log files."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Configure logging with rotation
    # Create a rotating file handler (max 10MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        os.path.join(save_dir, 'training.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')  # Simpler for console
    
    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def train_proc(args):
    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Set basic NCCL timeout to prevent timeouts during validation
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes timeout
    
    try:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        # Initialize process group
        dist.init_process_group(backend="nccl")
        
        if local_rank == 0:
            logging.info(f"[Rank {local_rank}] Starting distributed training with {world_size} GPUs")
    except Exception as e:
        logging.error(f"[Rank {local_rank}] Failed to initialize distributed training: {e}")
        raise

    # Set seeds for reproducibility across processes
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        if local_rank == 0:
            logging.info(f"[Rank {local_rank}] Loading dataset...")
        
        # Data
        if args.dataset_type == 'image':
            dataset = ContourDataset(
                label_file=os.path.join(args.data_path, args.csv_path),
                img_dir=args.data_path,
            )
        elif args.dataset_type == 'latent':
            dataset = LatentDataset(
                data_dir=args.latent_datapath,
                latent_size=args.latent_size
            )
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")
        
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        # Use a fixed generator for consistent splits across processes
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )

        sampler = distributed.DistributedSampler(train_ds, seed=42)
        val_sampler   = distributed.DistributedSampler(val_ds, shuffle=False, seed=42)

        # Reduce num_workers to prevent memory issues
        num_workers = min(args.num_workers, 4)  # Cap at 4 workers
        
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,  # Reduced from 16
            persistent_workers=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,  # Reduced from 16
            persistent_workers=True,
            drop_last=True
        )

        if local_rank == 0:
            logging.info(f"[Rank {local_rank}] Creating model...")
        
        # Model + (optional) Discriminator
        if args.dataset_type == 'latent':
            in_channels = args.latent_dim + args.contour_channels
            out_channels = args.latent_dim
            current_img_size = args.latent_size
        else: # image
            in_channels = 1 + args.contour_channels # image + contour
            out_channels = 1 # image
            current_img_size = args.img_size

        model = get_model(
            img_size=current_img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            time_dim=args.time_dim,
            pretrained_ckpt=args.encoder_ckpt
        ).to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        discriminator = None
        if "adv" in args.losses:
            discriminator = PatchGANDiscriminator(device=device).to(device)
            discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank)

        # Optimizers
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
        if discriminator is not None:
            optim_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d)
        else:
            optim_d = None

        # Optional checkpoint load
        if args.load_model:
            ckpt = torch.load(args.load_model, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            logging.info(f"[Rank {local_rank}] Loaded model epoch {ckpt.get('epoch','?')}")

        if local_rank == 0:
            logging.info(f"[Rank {local_rank}] Setting up diffusion...")
        
        # Diffusion setup
        diffusion = Diffusion(
            noise_step=args.noise_steps,
            img_size=current_img_size,
            device=device,
            schedule_name=args.noise_schedule
        )

        if local_rank == 0:
            logging.info(f"[Rank {local_rank}] Starting training...")
        
        # Training
        samples = ddpm_train(
            model=model,
            diffusion=diffusion,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            discriminator=discriminator,
            scaler=None,
            scheduler=scheduler,
            ema_model=None,
            metrics_callback=None
        )

        # Save EMA samples (rank 0 only)
        if dist.get_rank() == 0:
            grid = vutils.make_grid(samples, nrow=4, normalize=True)
            os.makedirs(args.save_dir, exist_ok=True)
            plt.imsave(
                os.path.join(args.save_dir, "ema_samples.png"),
                grid.permute(1, 2, 0).cpu().numpy()
            )
    except Exception as e:
        logging.error(f"[Rank {local_rank}] Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion Model Training")
    
    # --- Paths and Data ---
    parser.add_argument("--data_path", type=str, default="/hot/Yi-Kuan/Fibrosis", help="Path to the original image dataset directory.")
    parser.add_argument("--csv_path", type=str, default="small_label.csv", help="Path to the dataset CSV file, relative to data_path.")
    parser.add_argument("--latent_datapath", type=str, default="./data/latents_dataset", help="Path to the directory containing pre-computed latents.")
    parser.add_argument("--save_dir", type=str, default="model_runs/ldm_run_1", help="Path to save the model, logs, and samples.")
    parser.add_argument('--dataset_type', type=str, default='latent', choices=['image', 'latent'], help='Type of dataset to use for training. Should be "latent".')
    
    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=1000, help="Total number of training epochs.")
    parser.add_argument("--batch_size",  type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the U-Net optimizer.")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="Learning rate for the discriminator optimizer.")
    parser.add_argument("--save_interval", type=int, default=25, help="Epoch interval to save model checkpoints.")
    parser.add_argument("--metrics_interval", type=int, default=10, help="Epoch interval to compute validation metrics.")
    parser.add_argument("--early_stop_patience",type=int, default=10, help="Patience for early stopping based on validation loss.")
    
    # --- Model Hyperparameters ---
    parser.add_argument("--img_size", type=int, default=256, help="Size of the original, full-resolution image (for reference).")
    parser.add_argument("--vae_checkpoint", type=str, default="vae_checkpoint/vae_best.pth", help="Path to the trained VAE model checkpoint.")
    parser.add_argument("--latent_size", type=int, default=16, help="Spatial size of the latent space (e.g., 16x16).")
    parser.add_argument("--latent_dim", type=int, default=8, help="Number of channels in the latent space (from VAE).")
    parser.add_argument("--contour_channels", type=int, default=1, help="Number of channels for the contour condition.")
    parser.add_argument("--time_dim", type=int, default=256, help="Dimension of the time embedding in the U-Net.")
    parser.add_argument("--encoder_ckpt", type=str, default=None, help="Path to a pretrained encoder checkpoint (optional).")
    
    # --- Diffusion Hyperparameters ---
    parser.add_argument("--noise_steps", type=int, default=1000, help="Total number of steps in the diffusion process.")
    parser.add_argument("--noise_schedule", type=str, default='cosine', choices=['cosine', 'linear'], help="Noise schedule for the diffusion process.")

    # --- Loss Configuration ---
    parser.add_argument("--losses", type=lambda s: s.split(","), default=["mse"], help="Comma-separated list of losses to use (e.g., mse,lpips,adv).")
    parser.add_argument("--lambda_mse", type=float, default=1.0, help="Weight for the MSE loss term.")
    parser.add_argument("--lambda_lpips", type=float, default=10.0, help="Weight for the LPIPS perceptual loss term.")
    parser.add_argument("--lambda_adv", type=float, default=0.1, help="Weight for the adversarial loss term.")

    # --- Misc & Technical ---
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training.")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a full checkpoint to resume training.")
    parser.add_argument("--sample_batch_size",  type=int, default=16, help="Number of samples to generate for visualization.")
    parser.add_argument("--ema_decay", float, default=0.9999, help="Decay rate for the Exponential Moving Average of model weights.")
    parser.add_argument("--use_amp", action="store_true", help="Enable Automatic Mixed Precision (AMP) for training.")
    parser.add_argument("--no_sync_on_compute", action="store_true", help="Disable torchmetrics synchronization on each computation step.")

    # Filter out empty or whitespace-only arguments that can occur with multi-line shell commands.
    filtered_args = [arg for arg in sys.argv[1:] if arg.strip()]
    args = parser.parse_args(filtered_args)

    # Setup logging after parsing args
    setup_logging(args.save_dir)

    # Defer the main training process call
    train_proc(args)


if __name__ == '__main__':
    main()
