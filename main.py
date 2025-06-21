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
from models import get_model
from dataset import ContourDataset
from discriminator import PatchGANDiscriminator
from train_utils import train as ddpm_train, cosine_beta_schedule
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


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
            print(f"[Rank {local_rank}] Starting distributed training with {world_size} GPUs")
    except Exception as e:
        print(f"[Rank {local_rank}] Failed to initialize distributed training: {e}")
        raise

    # Set seeds for reproducibility across processes
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        if local_rank == 0:
            print(f"[Rank {local_rank}] Loading dataset...")
        
        # Data
        dataset = ContourDataset(args.csv_file, args.data_dir)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        # Use a fixed generator for consistent splits across processes
        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )

        train_sampler = distributed.DistributedSampler(train_ds, seed=42)
        val_sampler   = distributed.DistributedSampler(val_ds, shuffle=False, seed=42)

        # Reduce num_workers to prevent memory issues
        num_workers = min(args.num_workers, 4)  # Cap at 4 workers
        
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
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
            print(f"[Rank {local_rank}] Creating model...")
        
        # Model + (optional) Discriminator
        model = get_model(
            img_size=args.image_size,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            pretrained_ckpt = args.encoder_ckpt
        )
        if args.encoder_ckpt:
            state = torch.hub.load_state_dict_from_url(args.encoder_ckpt, map_location="cpu", check_hash=True)
            missing, _ = model.load_state_dict(state, strict=False)
            if local_rank == 0:
                print(f"[Info] Loaded encoder weights; missing keys: {missing}")
        model = model.to(device)
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
            print(f"[Rank {local_rank}] Loaded model epoch {ckpt.get('epoch','?')}")

        if local_rank == 0:
            print(f"[Rank {local_rank}] Setting up diffusion...")
        
        # Diffusion setup
        diffusion = Diffusion(
            noise_step=args.noise_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            img_size=args.image_size,
            device=device
        )
        diffusion.betas = cosine_beta_schedule(
            diffusion.noise_step
        ).to(device)
        diffusion.alpha     = 1.0 - diffusion.beta
        diffusion.alpha_hat = torch.cumprod(diffusion.alpha, dim=0)

        if local_rank == 0:
            print(f"[Rank {local_rank}] Starting training...")
        
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
        print(f"[Rank {local_rank}] Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="DDPM training")
    # Data & training hyperparameters
    parser.add_argument("--data_dir",    type=str, default="/hot/Yi-Kuan/Fibrosis")
    parser.add_argument("--csv_file",    type=str, default="/hot/Yi-Kuan/Fibrosis/label.csv")
    parser.add_argument("--save_dir",    type=str, default=".")
    parser.add_argument("--batch_size",  type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--noise_steps", type=int, default=1000)
    parser.add_argument("--beta_start",  type=float, default=1e-4)
    parser.add_argument("--beta_end",    type=float, default=0.02)

    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--lr_d",        type=float, default=1e-4)
    parser.add_argument("--epochs",      dest="num_epochs", type=int, default=1000)
    parser.add_argument("--encoder_ckpt",type=str, default=None)
    parser.add_argument("--save_interval",      type=int, default=30)
    parser.add_argument("--metrics_interval",   type=int, default=5,
                        help="Interval in epochs to compute realism metrics")
    parser.add_argument("--sample_batch_size",  type=int, default=16)
    parser.add_argument("--ema_decay",          type=float, default=0.9999)
    parser.add_argument("--early_stop_patience",type=int,   default=10)
    parser.add_argument("--use_amp",            action="store_true",
                        help="Enable mixed-precision training")
    parser.add_argument("--use_compile",        action="store_true",
                        help="Enable torch.compile() optimization")
    parser.add_argument("--in_channels", type=int, default=2)
    parser.add_argument("--out_channels",type=int, default=1)
    parser.add_argument("--no_sync_on_compute", action="store_true",
                        help="Disable torchmetrics sync on compute")
    
    # Loss configuration
    parser.add_argument(
        "--losses",
        type=lambda s: s.split(","),
        default=["mse"],
        help="comma-separated list: mse,dice,boundary,focal,adv"
    )
    parser.add_argument("--lambda_mse",      type=float, default=1.0)
    parser.add_argument("--lambda_dice",     type=float, default=0.0)
    parser.add_argument("--lambda_boundary", type=float, default=0.0)
    parser.add_argument("--lambda_focal",    type=float, default=0.0)
    parser.add_argument("--lambda_adv",      type=float, default=0.0)
    parser.add_argument("--lambda_lpips",    type=float, default=0.0,
                        help="Weight for LPIPS perceptual loss")

    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to checkpoint for fine-tuning")

    # Filter out empty or whitespace-only arguments that can occur with
    # multi-line shell commands.
    filtered_args = [arg for arg in sys.argv[1:] if arg.strip()]
    args = parser.parse_args(filtered_args)
    train_proc(args)


if __name__ == "__main__":
    main()
