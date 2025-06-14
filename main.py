import os
import argparse
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

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def train_proc(args):
    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    # Data
    dataset = ContourDataset(args.csv_file, args.data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_sampler = distributed.DistributedSampler(train_ds)
    val_sampler   = distributed.DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # Model + (optional) Discriminator
    model = get_model(
        args.arch,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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

    # Training
    samples = ddpm_train(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        args=args,
        discriminator=discriminator
    )

    # Save EMA samples (rank 0 only)
    if dist.get_rank() == 0:
        grid = vutils.make_grid(samples, nrow=4, normalize=True)
        os.makedirs(args.save_dir, exist_ok=True)
        plt.imsave(
            os.path.join(args.save_dir, "ema_samples.png"),
            grid.permute(1, 2, 0).cpu().numpy()
        )

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
    parser.add_argument("--arch",        type=str, default="unet2d",
                        choices=["unet2d", "swin_unet", "mask2former"])
    parser.add_argument("--encoder_ckpt",type=str, default=None)
    parser.add_argument("--save_interval",      type=int, default=30)
    parser.add_argument("--sample_batch_size",  type=int, default=16)
    parser.add_argument("--ema_decay",          type=float, default=0.9999)
    parser.add_argument("--early_stop_patience",type=int,   default=10)
    parser.add_argument("--use_amp",            action="store_true",
                        help="Enable mixed-precision training")
    parser.add_argument("--use_compile",        action="store_true",
                        help="Enable torch.compile() optimization")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels",type=int, default=1)
    
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

    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to checkpoint for fine-tuning")

    args = parser.parse_args()
    train_proc(args)


if __name__ == "__main__":
    main()
