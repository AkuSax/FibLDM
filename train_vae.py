# train_vae_advanced.py (with per-step logging)
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.utils import make_grid
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import lpips

from dataset import HDF5ImageDataset
from utils import EarlyStopper

# --- (NLayerDiscriminator class remains unchanged) ---
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE with advanced losses.")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="finetuned_vae_advanced")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_dataloader_workers", type=int, default=16)
    parser.add_argument("--log_every_epochs", type=int, default=5)
    parser.add_argument("--disc_start_epoch", type=int, default=10)
    parser.add_argument("--perceptual_weight", type=float, default=0.1)
    parser.add_argument("--adversarial_weight", type=float, default=0.05)
    parser.add_argument("--subset_fraction", type=float, default=None)
    # --- NEW ARGUMENT FOR PER-STEP LOGGING ---
    parser.add_argument("--log_every_steps", type=int, default=50, help="Log losses every N steps.")
    return parser.parse_args()

def main():
    args = parse_args()
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb", project_config=project_config)
    accelerator.init_trackers("vae-advanced-finetune", config=vars(args))

    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    discriminator = NLayerDiscriminator(input_nc=3, n_layers=3)
    perceptual_loss = lpips.LPIPS(net='vgg').to(accelerator.device)

    optimizer_g = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    full_dataset = HDF5ImageDataset(data_dir=args.data_dir, manifest_file='manifest_final.csv', image_size=256)

    if args.subset_fraction:
        if accelerator.is_main_process:
            print(f"Using a {args.subset_fraction*100:.1f}% random subset of the data.")
        subset_size = int(len(full_dataset) * args.subset_fraction)
        indices = torch.randperm(len(full_dataset))[:subset_size]
        dataset_to_split = Subset(full_dataset, indices)
    else:
        dataset_to_split = full_dataset

    if accelerator.is_main_process:
        print(f"Total dataset size for this run: {len(dataset_to_split)} images.")
        
    train_size = int(0.95 * len(dataset_to_split))
    val_size = len(dataset_to_split) - train_size
    train_dataset, val_dataset = random_split(dataset_to_split, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_dataloader_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.train_batch_size, num_workers=args.num_dataloader_workers)
    
    fixed_val_batch = next(iter(val_dataloader))

    vae, discriminator, perceptual_loss, optimizer_g, optimizer_d, train_dataloader, val_dataloader = accelerator.prepare(
        vae, discriminator, perceptual_loss, optimizer_g, optimizer_d, train_dataloader, val_dataloader
    )

    best_val_loss = float("inf")
    global_step = 0 # --- ADDED: Initialize global step counter ---

    for epoch in range(args.num_train_epochs):
        vae.train()
        discriminator.train()

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            images = batch
            
            posterior = vae.module.encode(images).latent_dist
            reconstructions = vae.module.decode(posterior.sample()).sample

            optimizer_g.zero_grad()
            
            recon_loss_mse = F.mse_loss(reconstructions, images)
            perceptual_loss_val = perceptual_loss(reconstructions, images).mean()
            kl_loss = posterior.kl().mean()
            g_loss = recon_loss_mse + args.perceptual_weight * perceptual_loss_val + 1e-6 * kl_loss

            if epoch >= args.disc_start_epoch:
                logits_fake = discriminator(reconstructions)
                adversarial_loss = -torch.mean(logits_fake)
                g_loss += args.adversarial_weight * adversarial_loss
            
            accelerator.backward(g_loss)
            optimizer_g.step()

            d_loss = None
            if epoch >= args.disc_start_epoch:
                optimizer_d.zero_grad()
                logits_real = discriminator(images.detach())
                logits_fake = discriminator(reconstructions.detach())
                loss_real = torch.mean(F.relu(1. - logits_real))
                loss_fake = torch.mean(F.relu(1. + logits_fake))
                d_loss = (loss_real + loss_fake) * 0.5
                accelerator.backward(d_loss)
                optimizer_d.step()
            
            # --- ADDED: Per-step logging ---
            if accelerator.is_main_process and global_step % args.log_every_steps == 0:
                log_dict = {
                    "g_loss": g_loss.item(),
                    "recon_mse": recon_loss_mse.item(),
                    "perceptual": perceptual_loss_val.item(),
                    "kl_div": kl_loss.item()
                }
                if d_loss is not None:
                    log_dict["d_loss"] = d_loss.item()
                accelerator.log(log_dict, step=global_step)
            
            global_step += 1 # --- ADDED: Increment global step ---

        if accelerator.is_main_process:
            vae.eval()
            # ... (validation loop remains the same) ...
            # ... (model saving and image logging remains the same) ...

    accelerator.end_training()

if __name__ == "__main__":
    main()