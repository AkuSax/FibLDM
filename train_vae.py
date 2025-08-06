# train_vae.py (With Checkpointing and Advanced Visualizations)
import os
import argparse
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.utils import save_image, make_grid
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from dataset import ImageDatasetForVAE
from utils import EarlyStopper

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a VAE on CT scan images using Accelerate.")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model ID.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with manifest.csv and images.")
    parser.add_argument("--output_dir", type=str, default="finetuned_vae", help="Directory to save the trained VAE.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size per GPU.")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Validation batch size per GPU.")
    parser.add_argument("--num_dataloader_workers", type=int, default=16, help="Number of workers for DataLoader.")
    parser.add_argument("--log_every_epochs", type=int, default=2, help="Log sample reconstructions every N epochs.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--image_size", type=int, default=256, help="Image resize dimension.")
    parser.add_argument("--subset_fraction", type=float, default=None, help="Use only a fraction of the dataset for quick testing (e.g., 0.1 for 10%).")
    parser.add_argument("--log_every_steps", type=int, default=100, help="Log training loss every N steps to W&B.")
    return parser.parse_args()

def main():
    args = parse_args()

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with="wandb",
        project_config=project_config,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="vae-ct-scan-finetune-accelerated",
            config=vars(args)
        )

    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    early_stopper = EarlyStopper(patience=args.early_stopping_patience, min_delta=1e-5)

    full_dataset = ImageDatasetForVAE(
        data_dir=args.data_dir,
        manifest_file='manifest.csv',
        image_size=args.image_size
    )

    if args.subset_fraction:
        if accelerator.is_main_process:
            print(f"Using a {args.subset_fraction*100:.1f}% subset of the data.")
        subset_size = int(len(full_dataset) * args.subset_fraction)
        indices = range(subset_size)
        full_dataset = Subset(full_dataset, indices)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    if accelerator.is_main_process:
        print(f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    train_dataloader = DataLoader(...)
    val_dataloader = DataLoader(...)

    # --- NEW: Get a fixed batch from the validation loader for consistent visualization ---
    fixed_val_batch = next(iter(val_dataloader))

    vae, optimizer, train_dataloader, val_dataloader = accelerator.prepare(...)

    best_val_loss = float("inf")

    accelerator.print("🚀 Starting VAE fine-tuning...")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        vae.train()
        train_loss = 0.0
        
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}")
        for batch in train_dataloader:
            images = batch
            optimizer.zero_grad()

            posterior = vae.module.encode(images).latent_dist
            latents = posterior.sample()
            reconstruction = vae.module.decode(latents).sample
            
            recon_loss = F.mse_loss(reconstruction, images, reduction="mean")
            kl_loss = posterior.kl().mean()
            loss = recon_loss + 1e-6 * kl_loss

            accelerator.backward(loss)
            optimizer.step()

            if accelerator.is_main_process:
                if global_step % args.log_every_steps == 0:
                    accelerator.log({"train_loss_step": loss.item()}, step=global_step)
            global_step += 1

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
        progress_bar.close()

        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch
                posterior = vae.module.encode(images).latent_dist
                latents = posterior.sample()
                reconstruction = vae.module.decode(latents).sample
                
                recon_loss = F.mse_loss(reconstruction, images, reduction="mean")
                kl_loss = posterior.kl().mean()
                loss = recon_loss + 1e-6 * kl_loss
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        
        avg_val_loss_gathered = accelerator.gather(torch.tensor(avg_val_loss).to(accelerator.device)).mean().item()

        if accelerator.is_main_process:
            accelerator.print(f"Epoch {epoch+1}: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss_gathered:.6f}")
            accelerator.log({
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss_gathered,
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        scheduler.step(avg_val_loss_gathered)

        # --- NEW: Checkpointing and Advanced Visual Logging ---
        if accelerator.is_main_process:
            # Save the model if the validation loss is the best we've seen so far.
            if avg_val_loss_gathered < best_val_loss:
                best_val_loss = avg_val_loss_gathered
                save_path = os.path.join(args.output_dir, "best_model")
                accelerator.save_state(save_path)
                accelerator.print(f"✅ New best model saved to {save_path} with validation loss: {best_val_loss:.6f}")

            if (epoch + 1) % args.log_every_epochs == 0:
                # 1. Log Reconstructions (Source vs. Generated)
                # Use the fixed batch and move it to the correct GPU
                sample_images = fixed_val_batch.to(accelerator.device)
                with torch.no_grad():
                    reconstruction = vae.module.decode(vae.module.encode(sample_images).latent_dist.sample()).sample
                
                # Create a grid with originals on top and reconstructions below
                comparison_grid = make_grid(torch.cat([sample_images, reconstruction]), nrow=len(sample_images))
                comparison_grid = (comparison_grid.clamp(-1, 1) + 1) / 2
                
                # 2. Log Interpolations
                start_image = sample_images[0:1] # First image in the batch
                end_image = sample_images[1:2]   # Second image in the batch
                
                with torch.no_grad():
                    z_start = vae.module.encode(start_image).latent_dist.sample()
                    z_end = vae.module.encode(end_image).latent_dist.sample()
                
                # Linearly interpolate between the two latent vectors
                alphas = torch.linspace(0, 1, steps=8, device=accelerator.device)
                interpolated_latents = torch.stack([(1 - alpha) * z_start + alpha * z_end for alpha in alphas])
                interpolated_latents = interpolated_latents.squeeze(1) # Remove extra dimension

                with torch.no_grad():
                    interpolated_images = vae.module.decode(interpolated_latents).sample

                # Create a grid of the interpolated images
                interpolation_grid = make_grid(interpolated_images, nrow=8)
                interpolation_grid = (interpolation_grid.clamp(-1, 1) + 1) / 2
                
                accelerator.get_tracker("wandb").log({
                    "reconstructions": wandb.Image(comparison_grid),
                    "interpolations": wandb.Image(interpolation_grid)
                })

            if early_stopper.early_stop(avg_val_loss_gathered):
                accelerator.print("Early stopping triggered.")
                break

    accelerator.print("🏁 Training finished. Saving final model...")
    
    unwrapped_vae = accelerator.unwrap_model(vae)
    unwrapped_vae.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    accelerator.end_training()

if __name__ == "__main__":
    main()