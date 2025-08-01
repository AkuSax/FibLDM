import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import wandb
import numpy as np
import argparse
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import LatentDataset
from utils import EarlyStopper

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a U-Net with LoRA.")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="The base model ID from Hugging Face.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the latent dataset.")
    parser.add_argument("--output_dir", type=str, default="lora_trained_unet", help="Directory to save the trained LoRA layers.")
    parser.add_argument("--lora_rank", type=int, default=16, help="The rank of the LoRA matrices.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The learning rate for the optimizer.")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--num_dataloader_workers", type=int, default=8, help="Number of workers for the DataLoader.")
    parser.add_argument("--eval_epochs", type=int, default=10, help="Generate sample images every N epochs.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Patience for early stopping.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    wandb.init(
        project="lora-ct-scan-finetune", # Name of your project
        config=args # Log all hyperparameters from the script
    )   
    
    images_output_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(images_output_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank*2,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj", "out_proj"],
    )
    unet = get_peft_model(unet, lora_config)
    text_encoder = get_peft_model(text_encoder, lora_config)
    
    unet.print_trainable_parameters()
    text_encoder.print_trainable_parameters()

    full_dataset = LatentDataset(data_dir=args.data_dir)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=args.num_dataloader_workers, 
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=False, 
        num_workers=args.num_dataloader_workers, 
        pin_memory=True
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * args.num_train_epochs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet.to(device)
    vae.to(device)
    scaler = GradScaler()

    early_stopper = EarlyStopper(patience=args.early_stopping_patience, min_delta=0.001)

    @torch.no_grad()
    def log_images(epoch):
        unet.eval()
        generator = torch.Generator(device=device).manual_seed(42)
        sample_size = 64
        eval_batch_size = 4
        latents = torch.randn(
            (eval_batch_size, unet.config.in_channels, sample_size, sample_size),
            device=device,
            generator=generator,
        )

        uncond_embeddings_shape = (eval_batch_size, 77, unet.config.cross_attention_dim)
        uncond_embeddings = torch.zeros(uncond_embeddings_shape, device=device)

        for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
            with autocast():
                noise_pred = unet(latents, t, encoder_hidden_states=uncond_embeddings, return_dict=False)[0]
            latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        with torch.no_grad():
            images = vae.decode(1 / 0.18215 * latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        
        save_image(images, os.path.join(images_output_dir, f"sample_epoch_{epoch:04d}.png"), nrow=eval_batch_size)
        wandb.log({"samples": wandb.Image(os.path.join(images_output_dir, f"sample_epoch_{epoch:04d}.png"))})        
        
        unet.train()

    # --- Training Loop ---
    for epoch in range(args.num_train_epochs):
        # -- Training Step --
        unet.train()
        train_loss = 0.0
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}/{args.num_train_epochs}")
        for step, batch in enumerate(train_dataloader):
            latents = batch[0].to(device)
            prompts = batch[1]
            # Tokenize and encode the prompts
            text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids.to(device)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(text_input_ids)[0]
            # Generate noise and add it to the latents
            noise = torch.randn(latents.shape).to(device)
            bs = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            uncond_embeddings = torch.zeros((bs, 77, unet.config.cross_attention_dim), device=device)
            
            optimizer.zero_grad()
            with autocast():
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(noise_pred.float(), noise.float())
            
            scaler.scale(loss).backward()
            total_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
        progress_bar.close()


        # -- Validation Step --
        unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                latents = batch[0].to(device)
                noise = torch.randn(latents.shape).to(device)
                bs = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                uncond_embeddings = torch.zeros((bs, 77, unet.config.cross_attention_dim), device=device)
                
                with autocast():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=uncond_embeddings, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch}: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        wandb.log({
            "total_gradient_norm": total_norm.item(),
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        # -- Early Stopping Check --
        if early_stopper.early_stop(avg_val_loss):
            print("Early stopping triggered.")
            break

        # -- Image Logging --
        if ((epoch) % args.eval_epochs == 0 or epoch == args.num_train_epochs - 1) and epoch > 0:
            print(f"Generating sample images at epoch {epoch}...")
            log_images(epoch)

    os.makedirs(args.output_dir, exist_ok=True)
    unet.save_pretrained(args.output_dir)

    print("\nTraining complete. LoRA layers saved to:", args.output_dir)
    print("Sample images saved in:", images_output_dir)

if __name__ == "__main__":
    main()