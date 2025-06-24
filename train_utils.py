# ddpm/train_utils.py

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from tqdm import tqdm
import math
import os
import csv
import logging
import torch.nn as nn

import lpips
import torchvision.utils as vutils
import pandas as pd

from metrics import RealismMetrics
from utils import EarlyStopper, EMA

# Latent-style LPIPS feature extractor
class LatentFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.net(x)

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
    optim_d=None,
):
    """Train the Latent Diffusion Model."""
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    
    torch.backends.cudnn.benchmark = True
    if args.use_compile:
        model = torch.compile(model)

    # Initialize LPIPS if needed
    lpips_loss = None
    if 'lpips' in args.losses:
        lpips_loss = lpips.LPIPS(net='alex').to(device)

    # Initialize latent-style LPIPS feature extractor if needed
    latent_lpips = None
    if hasattr(args, 'lambda_latent_lpips') and args.lambda_latent_lpips > 0:
        latent_lpips = LatentFeatureExtractor(in_channels=args.latent_dim).to(device)
        latent_lpips.eval()

    # --- VAE and EMA setup ---
    vae = None
    if is_main and hasattr(args, 'vae_checkpoint') and args.vae_checkpoint:
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

    # --- Validation Dataloader for Metrics ---
    val_img_loader = None
    if is_main:
        # This loader is for generating full-resolution images for visual metrics (FID, KID, etc.)
        # It's only needed on the main process.
        from dataset import ContourDataset # Re-import for validation
        # Load manifest to get available indices
        manifest_path = os.path.join('../data', "manifest.csv")
        manifest = pd.read_csv(manifest_path)
        available_indices = manifest['original_file'].index.tolist()
        val_img_dataset = ContourDataset(label_file=os.path.join('./data', args.csv_path), img_dir='./data', istransform=False)
        # Only use indices that exist in the manifest
        val_img_subset = torch.utils.data.Subset(val_img_dataset, available_indices[:args.sample_batch_size * 4])
        val_img_loader = torch.utils.data.DataLoader(val_img_subset, batch_size=args.sample_batch_size, shuffle=False)


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
        
        # Use a custom progress bar that doesn't spam logs
        if is_main:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=100)
        else:
            pbar = train_loader
        
        # Note: The dataloader now yields (latents, contours)
        for batch_idx, (latents, contour) in enumerate(pbar):
            if latents.size(0) == 0: continue
                
            latents = latents.to(device, non_blocking=True)
            contour = contour.to(device, non_blocking=True) # Contour is already downsampled by LatentDataset

            # Sample noise and noise the latents
            t = diffusion.sample_timesteps(latents.size(0)).to(device)
            x_t, noise = diffusion.noise_image(latents, t) # `noise_image` works on any tensor

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                pred_noise = model(x_t, t, contour)
                total_loss = 0
                loss_dict = {}

                # --- MSE Loss ---
                if 'mse' in args.losses:
                    loss_mse = F.mse_loss(pred_noise, noise)
                    total_loss += args.lambda_mse * loss_mse
                    loss_dict['mse'] = loss_mse.item()

                # --- Latent-style LPIPS Loss ---
                if latent_lpips is not None:
                    with torch.no_grad():
                        feat_pred = latent_lpips(pred_noise)
                        feat_target = latent_lpips(noise)
                    latent_lpips_loss = torch.mean((feat_pred - feat_target) ** 2)
                    total_loss += args.lambda_latent_lpips * latent_lpips_loss
                    loss_dict['latent_lpips'] = latent_lpips_loss.item()

                # --- Adversarial Loss (require predicting x0) ---
                if 'adv' in args.losses:
                    pred_x0 = diffusion.predict_start_from_noise(x_t, t, pred_noise)
                    if discriminator is not None:
                        d_fake_pred = discriminator(pred_x0)
                        loss_adv = -torch.mean(d_fake_pred)
                        total_loss += args.lambda_adv * loss_adv
                        loss_dict['adv'] = loss_adv.item()

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- Discriminator Training Step ---
            if 'adv' in args.losses and discriminator is not None and optim_d is not None:
                optim_d.zero_grad()
                with autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    pred_x0_detached = pred_x0.detach()
                    
                    d_real_pred = discriminator(latents)
                    d_fake_pred = discriminator(pred_x0_detached)

                    d_loss_real = torch.mean(F.relu(1. - d_real_pred))
                    d_loss_fake = torch.mean(F.relu(1. + d_fake_pred))
                    d_loss = (d_loss_real + d_loss_fake) / 2
                
                scaler.scale(d_loss).backward()
                scaler.step(optim_d)
                scaler.update()
                loss_dict['d_loss'] = d_loss.item()


            running_loss += total_loss.item()
            ema.update(model)
            
            # Update progress bar with current loss (only on main process)
            if is_main and batch_idx % 50 == 0:  # Update every 50 batches to reduce spam
                pbar.set_postfix(loss_dict)

        avg_loss = running_loss / len(train_loader)
        if is_main:
            logging.info(f"Epoch {epoch} | Train loss: {avg_loss:.6f}")

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} [VAL]", leave=False, ncols=100) if is_main else val_loader
            for latents, contour in val_pbar:
                latents = latents.to(device, non_blocking=True)
                contour = contour.to(device, non_blocking=True)

                t = diffusion.sample_timesteps(latents.size(0)).to(device)
                x_t, noise = diffusion.noise_image(latents, t)
                
                with autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    pred_noise = model(x_t, t, contour)
                    val_loss = F.mse_loss(pred_noise, noise) # Just use MSE for validation loss
                
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        if is_main:
            logging.info(f"Epoch {epoch} | Validation loss: {avg_val_loss:.6f}")
        
        # --- Early Stopping ---
        if is_main:
            if stopper.early_stop(avg_val_loss):
                logging.info("Early stopping triggered by validation loss.")
                break

        # --- Visual Debugging: Sample, Decode, and Save ---
        if is_main and epoch % args.save_interval == 0:
            if vae is None:
                logging.warning("VAE not available on main process, skipping debug sample generation.")
            else:
                with torch.no_grad():
                    # Get a batch of validation latents and (downsampled) contours
                    val_latents, val_contours_downsampled = next(iter(val_loader))
                    val_latents = val_latents.to(device)
                    val_contours_downsampled = val_contours_downsampled.to(device)

                    # --- START: ADD THIS DEBUGGING BLOCK ---
                    # This code will run only on the first visualization save to avoid spamming logs.
                    if epoch == args.save_interval:
                        print("\n" + "="*25 + " CONTOUR DEBUGGING " + "="*25)
                        debug_contour = val_contours_downsampled[0]
                        print(f"Shape of a single contour from val_loader: {debug_contour.shape}")
                        print(f"Data type: {debug_contour.dtype}")
                        print(f"Min value: {debug_contour.min():.4f}")
                        print(f"Max value: {debug_contour.max():.4f}")
                        print(f"Mean value: {debug_contour.mean():.4f}")
                        print(f"Unique values: {torch.unique(debug_contour)}")
                        debug_save_path = os.path.join(args.save_dir, 'debug_contour_from_loader.png')
                        vutils.save_image(debug_contour, debug_save_path)
                        print(f"Saved a sample raw contour to {debug_save_path}")
                        print("="*69 + "\n")
                    # --- END: ADD THIS DEBUGGING BLOCK ---

                    logging.info("Generating debug samples...")
                    # 1. Generate new clean latents with the LDM
                    ema.apply_shadow(model)
                    model.eval()
                    generated_latents = diffusion.sample(
                        model, 
                        n=val_latents.size(0), 
                        condition=val_contours_downsampled,
                        latent_dim=args.latent_dim
                    )
                    ema.restore(model)
                    
                    # 2. Decode both original and generated latents into images
                    # This shows a side-by-side of VAE reconstruction vs. LDM's output
                    original_recons = vae.decode(val_latents)
                    generated_images = vae.decode(generated_latents)
                    
                    # 3. Save for comparison
                    save_path = os.path.join(args.save_dir, 'epoch_samples')
                    os.makedirs(save_path, exist_ok=True)
                    
                    # New: Save a 3-row grid: originals | contours | generated
                    num_show = min(10, val_latents.size(0))
                    # Normalize all images to [0, 1] for visualization
                    originals = original_recons[:num_show].clamp(-1, 1) * 0.5 + 0.5
                    contours = val_contours_downsampled[:num_show].clamp(0, 1)
                    generated = generated_images[:num_show].clamp(-1, 1) * 0.5 + 0.5

                    def ensure_three_channels(x):
                        if x.shape[1] == 1:
                            return x.repeat(1, 3, 1, 1)
                        return x

                    originals = ensure_three_channels(originals)
                    contours = ensure_three_channels(contours)
                    generated = ensure_three_channels(generated)

                    # Stack as rows: originals, contours, generated
                    # Upsample contours and generated to match originals' size for visualization
                    contours = F.interpolate(contours, size=originals.shape[-2:], mode='bilinear', align_corners=False)
                    # Remove redundant interpolation for generated images - they're already at correct size
                    # generated = F.interpolate(generated, size=originals.shape[-2:], mode='nearest')
                    comparison_grid = torch.cat([originals, contours, generated], dim=0)
                    vutils.save_image(comparison_grid, os.path.join(save_path, f"comparison_{epoch:04d}.png"), nrow=num_show)
                    
                    logging.info(f"Saved decoded debug samples for epoch {epoch}.")

        # --- Metrics Calculation on Validation Set ---
        metric_results = {}
        if is_main and epoch % args.metrics_interval == 0:
            if vae is None or val_img_loader is None:
                logging.warning("VAE or validation image loader not available, skipping metrics calculation.")
            else:
                model.eval()
                local_real_images = []
                local_fake_samples_decoded = []
                manifest_path = os.path.join(args.latent_datapath, "manifest.csv")
                manifest = pd.read_csv(manifest_path)
                latent_dir = os.path.join(args.latent_datapath, "latents")
                contour_dir = os.path.join(args.latent_datapath, "contours")
                n_samples = min(args.sample_batch_size * 4, len(manifest))
                with torch.no_grad():
                    for i in range(n_samples):
                        latent = torch.load(os.path.join(latent_dir, f"{i}.pt")).to(device)
                        # Ensure latent is [1, C, H, W] before decoding
                        if latent.dim() == 3:
                            latent = latent.unsqueeze(0)
                        elif latent.dim() == 4 and latent.shape[0] == 1:
                            pass
                        else:
                            raise ValueError(f"Unexpected latent shape: {latent.shape}")
                        contour = torch.load(os.path.join(contour_dir, f"{i}.pt")).to(device)
                        # Decode the latent to get the real image
                        real_image = vae.decode(latent).squeeze(0)
                        # Generate a fake sample from the model
                        contour_downsampled = F.interpolate(contour.unsqueeze(0), size=(args.latent_size, args.latent_size), mode='nearest').squeeze(0)
                        fake_latent = diffusion.sample(model, 1, contour_downsampled.unsqueeze(0), fast_sampling=True, latent_dim=args.latent_dim).squeeze(0)
                        fake_image = vae.decode(fake_latent.unsqueeze(0)).squeeze(0)
                        local_real_images.append(real_image.cpu())
                        local_fake_samples_decoded.append(fake_image.cpu())
                if is_main:
                    all_real_images = torch.stack(local_real_images, dim=0)
                    all_fake_samples = torch.stack(local_fake_samples_decoded, dim=0)
                    if all_real_images.numel() > 0:
                        logging.info("Computing FID, KID, LPIPS, SSIM...")
                        # Ensure images are on the same device as metrics
                        all_real_images = all_real_images.to(device)
                        all_fake_samples = all_fake_samples.to(device)
                        metric_results = metrics.compute_metrics(all_real_images, all_fake_samples)
                        log_line = f"Epoch {epoch} Metrics | " + " | ".join([f"{k}: {v:.4f}" for k, v in metric_results.items()])
                        logging.info(log_line)
                        with open(metrics_csv, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([epoch, metric_results.get('fid', -1), metric_results.get('kid', -1), metric_results.get('lpips', -1), metric_results.get('ssim', -1), avg_loss])
                        # --- Model Checkpoint Saving ---
                        if epoch % args.save_interval == 0:
                            trained_models_dir = os.path.join(args.save_dir, 'trained_models')
                            os.makedirs(trained_models_dir, exist_ok=True)
                            checkpoint_path = os.path.join(trained_models_dir, f"model_epoch_{epoch:04d}.pth")
                            torch.save(model.state_dict(), checkpoint_path)
                            logging.info(f"Saved model checkpoint at {checkpoint_path}")
                    else:
                        logging.warning("No images were gathered for metric computation.")

        if scheduler:
            scheduler.step()

    # Final barrier to ensure all processes exit cleanly after training loop
    if dist.is_initialized():
        dist.barrier()
        
    print(f"Rank {dist.get_rank() if dist.is_initialized() else 0} finished training.")
    return model