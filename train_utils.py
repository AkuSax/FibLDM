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
    """Train the model."""
    # Check if this is the main process
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    # Disable DDP optimizer to avoid compatibility issues with torch.compile
    torch._dynamo.config.optimize_ddp = False
    
    # Use aot_eager backend which is more stable
    model = torch.compile(model, backend="aot_eager")

    ema = EMA(model, decay=args.ema_decay)
    stopper = EarlyStopper(patience=args.early_stop_patience, mode='min')
    if scaler is None:
        scaler = GradScaler()
    metrics = RealismMetrics(device=device)
    # Instantiate LPIPS model for loss calculation
    if 'lpips' in args.losses:
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    model.to(device)

    # Setup CSV logging
    metrics_csv = os.path.join(args.save_dir, 'metrics_log.csv')
    if is_main and not os.path.exists(metrics_csv):
        with open(metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'fid', 'kid', 'lpips', 'ssim', 'train_loss'])

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        for images, contour in pbar:
            # Ensure we have a reasonable batch size for distributed training
            if images.size(0) == 0:
                if is_main:
                    print(f"Warning: Empty batch encountered, skipping...")
                continue
                
            images = images.to(device, non_blocking=True)
            contour = contour.to(device, non_blocking=True)

            # Sample noise and prepare input
            t = diffusion.sample_timesteps(images.size(0)).to(device)
            x_t, noise = diffusion.noise_image(images, t)
            x_in = torch.cat((x_t, contour), dim=1)

            optimizer.zero_grad()

            # Forward (with optional AMP)
            if args.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    pred_noise = model(x_in, t)
                    # Compute weighted sum of requested losses
                    total_loss = torch.tensor(0.0, device=device, dtype=pred_noise.dtype)
                    for name in args.losses:
                        if name == "adv":
                            if discriminator is None:
                                raise RuntimeError("--losses adv requires a discriminator")
                            disc_pred = discriminator(pred_noise)
                            l = LOSS_REGISTRY[name](disc_pred)
                        elif name == "lpips":
                            # Reconstruct predicted x0
                            sqrt_alpha_hat = torch.sqrt(diffusion.alpha_hat[t])[:, None, None, None]
                            sqrt_one_minus_alpha_hat = torch.sqrt(1. - diffusion.alpha_hat[t])[:, None, None, None]
                            pred_x0 = (x_t - sqrt_one_minus_alpha_hat * pred_noise) / sqrt_alpha_hat
                            pred_x0 = torch.tanh(pred_x0)
                            # LPIPS loss between predicted x0 and original image
                            l = lpips_loss_fn(pred_x0, images).mean()
                        else:
                            l = LOSS_REGISTRY[name](pred_noise, noise)
                        total_loss = total_loss + getattr(args, f"lambda_{name}") * l
            else:
                pred_noise = model(x_in, t)
                total_loss = torch.tensor(0.0, device=device)
                for name in args.losses:
                    if name == "adv":
                        if discriminator is None:
                            raise RuntimeError("--losses adv requires a discriminator")
                        disc_pred = discriminator(pred_noise)
                        l = LOSS_REGISTRY[name](disc_pred)
                    elif name == "lpips":
                        # Reconstruct predicted x0
                        sqrt_alpha_hat = torch.sqrt(diffusion.alpha_hat[t])[:, None, None, None]
                        sqrt_one_minus_alpha_hat = torch.sqrt(1. - diffusion.alpha_hat[t])[:, None, None, None]
                        pred_x0 = (x_t - sqrt_one_minus_alpha_hat * pred_noise) / sqrt_alpha_hat
                        pred_x0 = torch.tanh(pred_x0)
                        # LPIPS loss between predicted x0 and original image
                        l = lpips_loss_fn(pred_x0, images).mean()
                    else:
                        l = LOSS_REGISTRY[name](pred_noise, noise)
                    total_loss = total_loss + getattr(args, f"lambda_{name}") * l

            # Backward & optimizer step
            if args.use_amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            running_loss += total_loss.item()

            # EMA update
            ema.update(model)

        avg_loss = running_loss / len(train_loader)
        if is_main:
            print(f"Epoch {epoch} | Train loss: {avg_loss:.6f}")

        # --- Visual Debugging: Save sample grid every save_interval epochs ---
        if is_main and epoch % args.save_interval == 0:
            with torch.no_grad():
                val_images, val_contours = next(iter(val_loader))
                val_images = val_images.to(device)
                val_contours = val_contours.to(device)
                debug_samples = diffusion.sample(model, val_images.size(0), val_contours)
                save_debug_samples(epoch, val_images, val_contours, debug_samples)
                print(f"Saved debug samples for epoch {epoch}.")

        # Compute realism metrics on validation set
        metric_results = {}
        if is_main and epoch % args.metrics_interval == 0:
            metrics.reset()
            model.eval()
            with torch.no_grad():
                for images, contours in val_loader:
                    images = images.to(device)
                    contours = contours.to(device)
                    # Generate samples for this batch
                    samples = diffusion.sample(model, images.size(0), contours)
                    # Update metrics batch by batch
                    metrics.fid.update(images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images, real=True)
                    metrics.fid.update(samples.repeat(1, 3, 1, 1) if samples.shape[1] == 1 else samples, real=False)

                    # For KID, create a new metric for each batch to avoid memory issues
                    subset_size = min(50, max(1, images.size(0)//2))  # Ensure subset_size is at least 1
                    batch_kid = KernelInceptionDistance(subset_size=subset_size, normalize=True).to(device)
                    batch_kid.update(images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images, real=True)
                    batch_kid.update(samples.repeat(1, 3, 1, 1) if samples.shape[1] == 1 else samples, real=False)
                    kid_mean, kid_std = batch_kid.compute()
                    if 'kid' not in metric_results:
                        metric_results['kid'] = []
                        metric_results['kid_std'] = []
                    metric_results['kid'].append(kid_mean.item())
                    metric_results['kid_std'].append(kid_std.item())

                    # LPIPS and SSIM (process in small batches)
                    batch_size = 8
                    for i in range(0, images.size(0), batch_size):
                        end_idx = min(i + batch_size, images.size(0))
                        
                        # First ensure inputs are in [0,1]
                        images_norm = torch.clamp(images[i:end_idx], 0, 1)
                        samples_norm = torch.clamp(samples[i:end_idx], 0, 1)
                        
                        # Convert to [-1,1] range for LPIPS
                        real_lpips = (2.0 * images_norm - 1.0)
                        fake_lpips = (2.0 * samples_norm - 1.0)
                        
                        # Add channels if needed
                        if real_lpips.shape[1] == 1:
                            real_lpips = real_lpips.repeat(1, 3, 1, 1)
                            fake_lpips = fake_lpips.repeat(1, 3, 1, 1)
                        
                        # Final clamp to ensure strict [-1,1] range
                        real_lpips = torch.clamp(real_lpips, -1.0, 1.0)
                        fake_lpips = torch.clamp(fake_lpips, -1.0, 1.0)
                        
                        # Update metrics
                        metrics.lpips.update(fake_lpips, real_lpips)
                        metrics.ssim.update(samples_norm, images_norm)  # SSIM expects [0,1]

                # Compute metrics
                try:
                    metric_results['fid'] = metrics.fid.compute().item()
                except Exception as e:
                    print(f"[metrics] FID computation error: {str(e)}")
                    metric_results['fid'] = float('inf')
                try:
                    metric_results['kid'] = np.mean(metric_results['kid']) if 'kid' in metric_results else float('inf')
                    metric_results['kid_std'] = np.mean(metric_results['kid_std']) if 'kid_std' in metric_results else float('inf')
                except Exception as e:
                    print(f"[metrics] KID computation error: {str(e)}")
                    metric_results['kid'] = float('inf')
                    metric_results['kid_std'] = float('inf')
                try:
                    metric_results['lpips'] = metrics.lpips.compute().item()
                except Exception as e:
                    print(f"[metrics] LPIPS computation error: {str(e)}")
                    metric_results['lpips'] = float('inf')
                try:
                    metric_results['ssim'] = metrics.ssim.compute().item()
                except Exception as e:
                    print(f"[metrics] SSIM computation error: {str(e)}")
                    metric_results['ssim'] = float('inf')

                print(f"Epoch {epoch} | Metrics:")
                for name, value in metric_results.items():
                    print(f"  {name}: {value:.4f}")
            model.train()

        # Live CSV logging
        if is_main and epoch % args.metrics_interval == 0:
            with open(metrics_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    metric_results.get('fid', ''),
                    metric_results.get('kid', ''),
                    metric_results.get('lpips', ''),
                    metric_results.get('ssim', ''),
                    avg_loss
                ])

        # Console logging of all statistics
        if is_main:
            print(f"Epoch {epoch} | Statistics:")
            print(f"  fid: {metric_results.get('fid', '')}")
            print(f"  kid: {metric_results.get('kid', '')}")
            print(f"  lpips: {metric_results.get('lpips', '')}")
            print(f"  ssim: {metric_results.get('ssim', '')}")
            print(f"  train_loss: {avg_loss}")

        # Save best model based on LPIPS
        if is_main and epoch % args.metrics_interval == 0 and 'lpips' in metric_results:
            current_lpips = metric_results.get('lpips', float('inf'))
            print(f"Epoch {epoch} | Val LPIPS: {current_lpips:.4f}  (best: {stopper.best_score:.4f})")

            # Save best model if LPIPS has improved
            if current_lpips < stopper.best_score:
                print(f"New best LPIPS score. Saving model...")
                torch.save({
                    "model_state":     model.state_dict(),
                    "ema_state":       ema.shadow,
                    "optimizer_state": optimizer.state_dict(),
                    "epoch":           epoch,
                    "best_lpips":      current_lpips,
                    "metrics":         metric_results
                }, "best_ddpm_model.pth")
                print("Saved new best model.")

            # Early stopping check
            if stopper.early_stop(current_lpips):
                print(f"Early stopping triggered at epoch {epoch} due to no improvement in LPIPS.")
                break

        # Periodic checkpoint
        if is_main and epoch % args.save_interval == 0:
            save_dir = "trained_models"
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(
                save_dir, f"ddpmv2_model_{epoch:03d}.pt"
            )
            torch.save({
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch":           epoch,
                "loss":            avg_loss,
                "metrics":         metric_results if epoch % args.metrics_interval == 0 else None
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        if scheduler:
            scheduler.step()

    # Final sampling with EMA weights
    if is_main:
        print("Loading EMA weights for sampling…")
        ema.apply_shadow(model)
        model.eval()
        
        # Get a single batch from the validation loader
        try:
            final_val_images, final_val_contours = next(iter(val_loader))
        except StopIteration:
            print("Validation loader is empty, cannot generate final samples.")
            return None

        # Determine the sample batch size, ensuring it's not larger than the actual batch
        sample_batch_size = min(args.sample_batch_size, final_val_contours.size(0))
        
        # Take the slice for sampling
        sample_contour = final_val_contours[:sample_batch_size].to(device)
        
        # Generate the final samples
        samples = diffusion.sample(model, sample_batch_size, sample_contour)
        ema.restore(model)
    else:
        samples = None

    return samples
