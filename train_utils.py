# ddpm/train_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import torch._dynamo
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import math
import os
from ddpm.losses import LOSS_REGISTRY
from metrics import RealismMetrics
import csv
from utils import EarlyStopper, EMA, save_debug_samples
from metrics import KernelInceptionDistance
import lpips
import torchvision.utils as vutils


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.FloatTensor:
    steps = timesteps
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999)


def evaluate_iou(model, dataloader, device, diffusion):
    """Evaluate IoU on validation set."""
    model.eval()
    ious = []
    with torch.no_grad():
        # Get the underlying model if wrapped in DDP
        if hasattr(model, 'module'):
            model = model.module
            
        # Create progress bar for overall validation
        total_batches = len(dataloader)
        pbar = tqdm(dataloader, desc="Validating", leave=True)
        
        for batch_idx, (images, contours) in enumerate(pbar):
            images = images.to(device)
            contours = contours.to(device)
            
            # Sample from model using fast sampling for validation
            samples = diffusion.sample(model, images.size(0), contours, fast_sampling=True)
            
            # Convert to binary masks
            pred_masks = (samples > 0.5).float()
            true_masks = (contours > 0.5).float()
            
            # Calculate IoU
            intersection = (pred_masks * true_masks).sum(dim=(1, 2, 3))
            union = pred_masks.sum(dim=(1, 2, 3)) + true_masks.sum(dim=(1, 2, 3)) - intersection
            batch_iou = (intersection / (union + 1e-6)).mean().item()
            ious.append(batch_iou)
            
            # Update progress bar with current IoU
            pbar.set_postfix({
                'batch': f'{batch_idx + 1}/{total_batches}',
                'current_iou': f'{batch_iou:.4f}',
                'avg_iou': f'{sum(ious)/len(ious):.4f}'
            })
            
    return sum(ious) / len(ious)


def train(
    model,
    diffusion,
    optimizer,
    train_loader,
    val_loader,
    device,
    args,
    scaler=None,
    scheduler=None,
    ema_model=None,
    metrics_callback=None,
    discriminator=None,
):
    """Train the Latent Diffusion Model."""
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    
    torch.backends.cudnn.benchmark = True
    if args.use_compile:
        model = torch.compile(model)

    # --- VAE and EMA setup ---
    vae = None
    if is_main:
        # VAE is only needed on the main process for decoding samples for visualization
        print("Loading VAE for decoding...")
        from autoencoder import VAE # Local import
        vae = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
        # You must provide the path to your trained VAE checkpoint
        vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
        vae.eval()
        print("VAE loaded.")

    ema = EMA(model, decay=args.ema_decay)
    stopper = EarlyStopper(patience=args.early_stop_patience, mode='min')
    if scaler is None:
        scaler = GradScaler(enabled=args.use_amp)
    metrics = RealismMetrics(device=device, sync_on_compute=not args.no_sync_on_compute)

    # Setup CSV logging
    metrics_csv = os.path.join(args.save_dir, 'metrics_log.csv')
    if is_main and not os.path.exists(metrics_csv):
        with open(metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            # LPIPS is now a validation metric, not a loss term during latent training
            writer.writerow(['epoch', 'fid', 'kid', 'lpips', 'ssim', 'train_loss'])

    for epoch in range(args.num_epochs):
        model.train()
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        
        # Note: The dataloader now yields (latents, contours)
        for latents, contour in pbar:
            if latents.size(0) == 0: continue
                
            latents = latents.to(device, non_blocking=True)
            contour = contour.to(device, non_blocking=True) # Contour is already downsampled by LatentDataset

            # Sample noise and noise the latents
            t = diffusion.sample_timesteps(latents.size(0)).to(device)
            x_t, noise = diffusion.noise_image(latents, t) # `noise_image` works on any tensor
            x_in = torch.cat((x_t, contour), dim=1)

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                pred_noise = model(x_in, t)
                # The primary loss is now MSE between the predicted and actual noise in the latent space
                total_loss = F.mse_loss(pred_noise, noise)
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()
            ema.update(model)

        avg_loss = running_loss / len(train_loader)
        if is_main:
            print(f"Epoch {epoch} | Train loss: {avg_loss:.6f}")

        # --- Visual Debugging: Sample, Decode, and Save ---
        if is_main and epoch % args.save_interval == 0:
            if vae is None:
                print("Warning: VAE not available on main process, skipping debug sample generation.")
            else:
                with torch.no_grad():
                    # Get a batch of validation latents and (downsampled) contours
                    val_latents, val_contours_downsampled = next(iter(val_loader))
                    val_latents = val_latents.to(device)
                    val_contours_downsampled = val_contours_downsampled.to(device)

                    print("Generating debug samples...")
                    # 1. Generate new clean latents with the LDM
                    ema.ema_model.eval()
                    generated_latents = diffusion.sample(
                        ema.ema_model, 
                        n=val_latents.size(0), 
                        condition=val_contours_downsampled
                    )
                    
                    # 2. Decode both original and generated latents into images
                    # This shows a side-by-side of VAE reconstruction vs. LDM's output
                    original_recons = vae.decode(val_latents)
                    generated_images = vae.decode(generated_latents)
                    
                    # 3. Save for comparison
                    save_path = os.path.join(args.save_dir, 'epoch_samples')
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Save a grid comparing VAE reconstruction to LDM's output
                    comparison_grid = torch.cat([original_recons, generated_images])
                    vutils.save_image(comparison_grid, os.path.join(save_path, f"comparison_{epoch:04d}.png"), nrow=val_latents.size(0))
                    
                    print(f"Saved decoded debug samples for epoch {epoch}.")

        # --- Metrics Calculation on Validation Set ---
        metric_results = {}
        if is_main and epoch % args.metrics_interval == 0:
            if vae is None:
                print("Warning: VAE not available, skipping metrics calculation.")
            else:
                model.eval()
                
                # For metrics, we need to compare generated images to the *true* original images.
                # We need a separate loader for the original, high-resolution image dataset.
                from dataset import ContourDataset # Re-import for validation
                val_img_dataset = ContourDataset(csv_file=os.path.join(args.data_path, args.csv_path), img_dir=args.data_path, istransform=False)
                # Important: Use a DistributedSampler for the validation image loader as well
                val_img_sampler = torch.utils.data.distributed.DistributedSampler(val_img_dataset, shuffle=False)
                val_img_loader = torch.utils.data.DataLoader(val_img_dataset, batch_size=args.sample_batch_size, sampler=val_img_sampler)

                local_real_images = []
                local_fake_samples_decoded = []

                with torch.no_grad():
                    val_pbar = tqdm(val_img_loader, desc=f"Epoch {epoch} Calculating Metrics", disable=not is_main)
                    for images, contours in val_pbar:
                        images = images.to(device)
                        contours = contours.to(device)
                        
                        contours_downsampled = F.interpolate(contours, size=(args.latent_size, args.latent_size), mode='nearest')

                        # Generate latents from contour condition and decode them to images
                        samples_latent = diffusion.sample(model, images.size(0), contours_downsampled, fast_sampling=True)
                        samples_decoded = vae.decode(samples_latent)

                        local_real_images.append(images.cpu())
                        local_fake_samples_decoded.append(samples_decoded.cpu())
                
                # Gather results from all processes
                gathered_real = [None for _ in range(dist.get_world_size())]
                gathered_fake = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_real, local_real_images)
                dist.all_gather_object(gathered_fake, local_fake_samples)
                
                if is_main:
                    all_real_images = [item for sublist in gathered_real for item in sublist]
                    all_fake_samples = [item for sublist in gathered_fake for item in sublist]
                    
                    if all_real_images:
                        all_real_images = torch.cat(all_real_images, dim=0)
                        all_fake_samples = torch.cat(all_fake_samples, dim=0)

                        print(f"Calculating realism metrics on {len(all_real_images)} images...")
                        metric_results = metrics.compute(all_real_images, all_fake_samples)
                        print(f"Epoch {epoch} Metrics: {metric_results}")
                    else:
                        print("No images processed for metrics, skipping.")


        # --- Logging and Early Stopping (Main Process Only) ---
        if is_main:
            # Log metrics to CSV
            with open(metrics_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    metric_results.get('fid', 'N/A'),
                    metric_results.get('kid', 'N/A'),
                    metric_results.get('lpips', 'N/A'),
                    metric_results.get('ssim', 'N/A'),
                    avg_loss
                ])
            
            # Check for early stopping
            validation_metric = metric_results.get('lpips', float('inf')) # Using LPIPS as stopping criterion
            if stopper.early_stop(validation_metric):
                print(f"Early stopping triggered at epoch {epoch} based on validation LPIPS.")
                break # Exit training loop
    
    # Final barrier to ensure all processes exit cleanly after training loop
    if dist.is_initialized():
        dist.barrier()
        
    print(f"Rank {dist.get_rank() if dist.is_initialized() else 0} finished training.")
    return model


def monitor_gpu_memory(device, stage=""):
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
        print(f"[{stage}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        return allocated, reserved
    return 0, 0