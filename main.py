import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from DDPM import Diffusion
from model import UNet2D
from utils import save_images
from dataset import ContourDataset
from train_utils import train as ddpm_train, cosine_beta_schedule
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def train(args):
    # DDP process group header
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize process group
    dist.init_process_group(backend="nccl")

    # Dataset + DistributedSampler + DataLoader
    dataset = ContourDataset(args.csv_file, args.data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_sampler = distributed.DistributedSampler(train_ds)
    val_sampler = distributed.DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # Model + wrap in DDP
    model = UNet2D(device=device).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Optional checkpoint loading
    if args.load_model:
        ckpt = torch.load(args.load_model, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        print(f"[Rank {local_rank}] Loaded checkpoint epoch {ckpt.get('epoch','?')} IoU {ckpt.get('best_iou','?')}")

    # Diffusion + cosine schedule override
    diffusion = Diffusion(
        noise_step=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=args.image_size,
        device=device
    )
    diffusion.betas = cosine_beta_schedule(diffusion.noise_step).to(device)
    diffusion.alpha = 1.0 - diffusion.beta
    diffusion.alpha_hat = torch.cumprod(diffusion.alpha, dim=0)

    # Run training (EMA, early-stop, checkpointing)
    samples = ddpm_train(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        ema_decay=0.9999,
        early_stop_patience=10,
        sample_batch_size=args.batch_size
    )

    # Save final EMA samples (only on rank 0)
    grid = vutils.make_grid(samples, nrow=4, normalize=True)
    plt.imsave(os.path.join(args.save_dir, "ema_samples.png"), grid.permute(1,2,0).cpu().numpy())

    # Clean up DDP
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='DDPM multi-GPU training')
    parser.add_argument('--device',     type=str,   default='cuda')
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--epochs',     type=int,   default=1000)
    parser.add_argument('--image_size', type=int,   default=256)
    parser.add_argument('--save_dir',   type=str,   default='.')
    parser.add_argument('--data_dir',   type=str,   default='/hot/Yi-Kuan/Fibrosis')
    parser.add_argument('--csv_file',   type=str,   default='/hot/Yi-Kuan/Fibrosis/label.csv')
    parser.add_argument('--batch_size', type=int,   default=80)
    parser.add_argument('--load_model', type=str,   default=None)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
