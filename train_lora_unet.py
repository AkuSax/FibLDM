import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, random_split
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torchvision.utils import make_grid
import wandb

from dataset import LatentDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a U-Net with LoRA for CT scan generation.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with manifest, latents folder, etc.")
    parser.add_argument("--vae_model_path", type=str, required=True, help="Path to the fine-tuned VAE model directory.")
    parser.add_argument("--output_dir", type=str, default="lora_trained_unet", help="Directory to save the trained LoRA layers.")
    parser.add_argument("--lora_rank", type=int, default=32, help="The rank of the LoRA matrices.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size per GPU.")
    parser.add_argument("--num_dataloader_workers", type=int, default=16, help="Number of workers for DataLoader.")
    parser.add_argument("--log_every_epochs", type=int, default=5, help="Generate and log sample images every N epochs.")
    parser.add_argument("--log_every_steps", type=int, default=100, help="Log training loss every N steps to W&B.")
    return parser.parse_args()
    return parser.parse_args()

def main():
    args = parse_args()
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb", project_config=project_config)

    if accelerator.is_main_process:
        accelerator.init_trackers(project_name="lora-ct-scan-finetune", config=vars(args))

    model_id = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path)

    lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet = get_peft_model(unet, lora_config)
    if accelerator.is_main_process:
        unet.print_trainable_parameters()

    full_dataset = LatentDataset(data_dir=args.data_dir, manifest_file='manifest_balanced.csv')
    
    train_dataloader = DataLoader(
        full_dataset, 
        shuffle=True, # Shuffle the data each epoch
        batch_size=args.train_batch_size, 
        num_workers=args.num_dataloader_workers, 
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    unet, text_encoder, vae, optimizer, train_dataloader = accelerator.prepare(
        unet, text_encoder, vae, optimizer, train_dataloader
    )
    
    vae.eval()
    text_encoder.eval()

    accelerator.print("🚀 Starting LoRA fine-tuning on balanced dataset...")
    for epoch in range(args.num_train_epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}")
        
        for step, (latents, prompts) in enumerate(train_dataloader):
            optimizer.zero_grad()
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids.to(accelerator.device)
            
            with torch.no_grad():
                encoder_hidden_states = text_encoder(text_input_ids)[0]
            
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            optimizer.step()

            if accelerator.is_main_process:
                if global_step % args.log_every_steps == 0:
                    accelerator.log(
                        {"train_loss_step": loss.item()},
                        step=global_step
                    )
            global_step += 1

            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
        progress_bar.close()
        accelerator.log({"train_loss_epoch": loss.item()}, step=epoch)

        if accelerator.is_main_process and (epoch + 1) % args.log_every_epochs == 0:
            unet.eval()
            with torch.no_grad():
                prompt = "A transverse lung CT scan of a fibrosis lung, slice 150"
                text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_input_ids = text_inputs.input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(text_input_ids)[0]
                latents = torch.randn(1, unet.module.config.in_channels, 32, 32, device=accelerator.device)
                for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
                    noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
                    latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
                image = vae.module.decode(latents / vae.module.config.scaling_factor).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                accelerator.get_tracker("wandb").log({"sample_image": wandb.Image(image)})
            unet.train()

    accelerator.print("🏁 Training finished. Saving final LoRA layers...")
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(args.output_dir)
    accelerator.end_training()

if __name__ == "__main__":
    main()