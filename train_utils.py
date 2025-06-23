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

import lpips
import torchvision.utils as vutils

from metrics import RealismMetrics
from utils import EarlyStopper, EMA

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
        # Create a subset of the val_ds for faster metric calculation
        val_img_dataset = ContourDataset(label_file=os.path.join(args.data_path, args.csv_path), img_dir=args.data_path, istransform=False)
        
        # We need a way to select the same subset of images for validation metrics consistently
        # Here we'll just use the first N samples for simplicity
        val_indices = list(range(len(val_loader.dataset))) # WARNING: This gets indices from the LATENT val set
        val_img_subset = torch.utils.data.Subset(val_img_dataset, val_indices[:args.sample_batch_size * 4]) # Limit to a few batches
        
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
            x_in = torch.cat((x_t, contour), dim=1)

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                pred_noise = model(x_in, t)
                
                total_loss = 0
                loss_dict = {}

                # --- MSE Loss ---
                if 'mse' in args.losses:
                    loss_mse = F.mse_loss(pred_noise, noise)
                    total_loss += args.lambda_mse * loss_mse
                    loss_dict['mse'] = loss_mse.item()

                # --- LPIPS and Adversarial Losses (require predicting x0) ---
                if 'lpips' in args.losses or 'adv' in args.losses:
                    pred_x0 = diffusion.predict_start_from_noise(x_t, t, pred_noise)

                    # --- LPIPS Loss ---
                    if 'lpips' in args.losses and lpips_loss is not None:
                        # LPIPS expects images in [-1, 1] range, which latents are.
                        loss_lpips = lpips_loss(pred_x0, latents).mean()
                        total_loss += args.lambda_lpips * loss_lpips
                        loss_dict['lpips'] = loss_lpips.item()
                    
                    # --- Adversarial Loss (Generator part) ---
                    if 'adv' in args.losses and discriminator is not None:
                        # We want the discriminator to think the generated images are real
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
                x_in = torch.cat((x_t, contour), dim=1)
                
                with autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    pred_noise = model(x_in, t)
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

                    logging.info("Generating debug samples...")
                    # 1. Generate new clean latents with the LDM
                    ema.ema_model.eval()
                    generated_latents = diffusion.sample(
                        ema.ema_model, 
                        n=val_latents.size(0), 
                        condition=val_contours_downsampled,
                        latent_dim=args.latent_dim
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

                with torch.no_grad():
                    # Use a simple progress indicator instead of verbose tqdm
                    if is_main:
                        logging.info(f"Epoch {epoch}: Calculating metrics...")
                    
                    for images, contours in val_img_loader:
                        images = images.to(device)
                        contours = contours.to(device)
                        
                        contours_downsampled = F.interpolate(contours, size=(args.latent_size, args.latent_size), mode='nearest')

                        # Generate latents from contour condition and decode them to images
                        samples_latent = diffusion.sample(model, images.size(0), contours_downsampled, fast_sampling=True, latent_dim=args.latent_dim)
                        samples_decoded = vae.decode(samples_latent)

                        local_real_images.append(images.cpu())
                        local_fake_samples_decoded.append(samples_decoded.cpu())
                
                # This part doesn't need DDP gathering since it only runs on main process
                if is_main:
                    all_real_images = torch.cat(local_real_images, dim=0)
                    all_fake_samples = torch.cat(local_fake_samples_decoded, dim=0)
                    
                    if all_real_images.numel() > 0:
                        logging.info("Computing FID, KID, LPIPS, SSIM...")
                        metric_results = metrics.compute(all_fake_samples, all_real_images)
                        
                        # Log metrics to console and CSV
                        log_line = f"Epoch {epoch} Metrics | " + " | ".join([f"{k}: {v:.4f}" for k, v in metric_results.items()])
                        logging.info(log_line)
                        
                        with open(metrics_csv, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([epoch, metric_results.get('fid', -1), metric_results.get('kid', -1), metric_results.get('lpips', -1), metric_results.get('ssim', -1), avg_loss])
                    else:
                        logging.warning("No images were gathered for metric computation.")

        if scheduler:
            scheduler.step()

    # Final barrier to ensure all processes exit cleanly after training loop
    if dist.is_initialized():
        dist.barrier()
        
    print(f"Rank {dist.get_rank() if dist.is_initialized() else 0} finished training.")
    return model