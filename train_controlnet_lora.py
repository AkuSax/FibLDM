# train_controlnet_lora.py
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torchvision.utils import make_grid, save_image
import wandb

from dataset import ControlNetLatentDataset

def collate_fn(examples):
    """Custom collate function to filter out None values from the dataset."""
    examples = [e for e in examples if e is not None]
    if len(examples) == 0:
        return None
    latents, masks, prompts = zip(*examples)
    return torch.stack(latents), torch.stack(masks), prompts

def main():
    parser = argparse.ArgumentParser(description="Train ControlNet + LoRA for CT Scan Generation.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--vae_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="controlnet_lora_model")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_dataloader_workers", type=int, default=16)
    parser.add_argument("--log_every_epochs", type=int, default=1, help="Generate sample images every N epochs.")
    parser.add_argument("--log_every_steps", type=int, default=100, help="Log training loss every N steps.")
    args = parser.parse_args()

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb", project_config=project_config)
    accelerator.init_trackers("controlnet-lora-ct-scan", config=vars(args))

    model_id = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path)
    controlnet = ControlNetModel.from_unet(unet)

    lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet = get_peft_model(unet, lora_config)
    
    full_dataset = ControlNetLatentDataset(data_dir=args.data_dir, manifest_file='manifest_controlnet_balanced.csv')
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=args.num_dataloader_workers)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=args.num_dataloader_workers)
    
    fixed_val_batch = None
    if accelerator.is_main_process and val_size > 0:
        try:
            fixed_val_batch = next(iter(val_dataloader))
        except (StopIteration, TypeError):
            fixed_val_batch = None

    params_to_train = list(controlnet.parameters()) + list(unet.parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=args.learning_rate)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    controlnet, unet, text_encoder, vae, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        controlnet, unet, text_encoder, vae, optimizer, train_dataloader, val_dataloader
    )
    
    vae.eval()
    text_encoder.eval()

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.num_train_epochs):
        controlnet.train()
        unet.train()
        train_loss = 0.0
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(train_dataloader):
            if batch is None: continue
            latents, masks, prompts = batch
            optimizer.zero_grad()
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            with torch.no_grad():
                text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                encoder_hidden_states = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]
            
            down_block_res_samples, mid_block_res_sample = controlnet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=masks, return_dict=False)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample).sample
            
            loss = F.mse_loss(noise_pred.float(), noise.float())
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params_to_clip = list(accelerator.unwrap_model(controlnet).parameters()) + list(accelerator.unwrap_model(unet).parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
                accelerator.log({"gradient_norm": grad_norm.item()}, step=global_step)

            optimizer.step()
            train_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

            if accelerator.is_main_process and global_step % args.log_every_steps == 0:
                accelerator.log({"train_loss_step": loss.item()}, step=global_step)
            global_step += 1
        
        controlnet.eval()
        unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, disable=not accelerator.is_local_main_process, desc="Validating"):
                if batch is None: continue
                latents, masks, prompts = batch
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                with torch.no_grad():
                    text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                    encoder_hidden_states = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]
                down_block_res_samples, mid_block_res_sample = controlnet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=masks, return_dict=False)
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        accelerator.log({"avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss}, step=epoch)

        if accelerator.is_main_process:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(args.output_dir, "best_model")
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_controlnet = accelerator.unwrap_model(controlnet)
                unwrapped_unet.save_pretrained(os.path.join(save_path, "unet_lora"))
                unwrapped_controlnet.save_pretrained(os.path.join(save_path, "controlnet"))
                accelerator.print(f"✅ New best model saved to {save_path} with validation loss: {best_val_loss:.6f}")

            if (epoch + 1) % args.log_every_epochs == 0 and fixed_val_batch is not None:
                with torch.no_grad():
                    val_latents, val_masks, val_prompts = fixed_val_batch
                    num_samples = 4
                    sample_masks = val_masks[0:num_samples].to(accelerator.device)
                    
                    prompts = ["A transverse lung CT scan of a fibrosis lung, slice 150"] * num_samples
                    text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
                    encoder_hidden_states = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]
                    
                    latents = torch.randn(num_samples, unet.module.config.in_channels, 32, 32, device=accelerator.device)
                    
                    for t in tqdm(noise_scheduler.timesteps, desc="Generating Samples"):
                        down_block_res_samples, mid_block_res_sample = controlnet(latents, t, encoder_hidden_states=encoder_hidden_states, controlnet_cond=sample_masks, return_dict=False)
                        noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_states, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample).sample
                        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

                    images = vae.module.decode(latents / vae.module.config.scaling_factor).sample
                    
                    masks_vis = F.interpolate(sample_masks, size=(256, 256), mode='nearest')
                    
                    grid = make_grid(torch.cat([(images / 2 + 0.5).clamp(0, 1), masks_vis]), nrow=num_samples)
                    
                    # Save the grid to a file
                    samples_dir = os.path.join(args.output_dir, "samples")
                    os.makedirs(samples_dir, exist_ok=True)
                    save_image(grid, os.path.join(samples_dir, f"epoch_{epoch+1:04d}.png"))

                    accelerator.get_tracker("wandb").log({
                        "validation_samples": wandb.Image(grid, caption=f"Epoch {epoch+1}: Top - Generated, Bottom - Mask")
                    })

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save the final models
        final_save_path = os.path.join(args.output_dir, "final_model")
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        unwrapped_unet.save_pretrained(os.path.join(final_save_path, "unet_lora"))
        unwrapped_controlnet.save_pretrained(os.path.join(final_save_path, "controlnet"))
        print(f"✅ Final models saved to {final_save_path}")

if __name__ == "__main__":
    main()
