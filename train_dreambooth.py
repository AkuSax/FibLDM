# train_dreambooth.py
import argparse
import hashlib
import itertools
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import cv2
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel
import wandb
import multiprocessing as mp

logger = get_logger(__name__)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

# --- Custom Dataset for DreamBooth with Masks ---
class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance images and masks for DreamBooth training.
    It reads a directory of instance images and a manifest to find corresponding masks.
    """
    def __init__(self, instance_data_root, instance_prompt, tokenizer, manifest_path, data_dir, size=256):
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance images root {self.instance_data_root} doesn't exist.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images
        self.size = size
        self.tokenizer = tokenizer

        print(f"Loading manifest from {manifest_path} to find masks.")
        manifest = pd.read_csv(manifest_path)
        self.mask_map = {row['original_file']: row['mask_file'] for _, row in manifest.iterrows()}
        self.masks_dir = os.path.join(data_dir, 'masks')
        print(f"Found {len(self.mask_map)} entries in mask map.")

        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=True)
        ])


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        
        # Load Instance Image
        instance_image = cv2.imread(str(instance_image_path), cv2.IMREAD_GRAYSCALE)
        instance_image = cv2.resize(instance_image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        example["instance_images"] = self.image_transforms(instance_image).repeat(3, 1, 1)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        # Load Corresponding Mask
        image_filename = os.path.basename(instance_image_path)
        if image_filename in self.mask_map:
            mask_filename = self.mask_map[image_filename]
            mask_path = os.path.join(self.masks_dir, mask_filename)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            example["instance_masks"] = self.mask_transform(mask_img)
        else:
            # If no mask is found, return a black mask as a fallback
            print(f"Warning: No mask found for {image_filename}. Using a black mask.")
            example["instance_masks"] = torch.zeros(1, self.size, self.size)
            
        return example

def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth training script with ControlNet.")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the pretrained Stable Diffusion model (your fine-tuned one).")
    parser.add_argument("--pretrained_vae_path", type=str, required=True, help="Path to the fine-tuned VAE model.")
    parser.add_argument("--pretrained_controlnet_path", type=str, required=True, help="Path to the fine-tuned ControlNet model.")
    parser.add_argument("--instance_data_dir", type=str, required=True, help="A folder containing the training data of instance images.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the main data directory containing manifest and masks folder.")
    parser.add_argument("--output_dir", type=str, default="dreambooth-model", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--instance_prompt", type=str, required=True, help="The prompt with identifier specifying the instance, e.g., 'a photo of sks fibrotic lung'")
    parser.add_argument("--class_prompt", type=str, default="a transverse lung CT scan", help="The prompt to generate images for class regularization.")
    parser.add_argument("--with_prior_preservation", default=True, action="store_true", help="Flag to add prior preservation loss.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument("--num_class_images", type=int, default=100, help="Minimal class images for prior preservation loss.")
    parser.add_argument("--resolution", type=int, default=256, help="The resolution for input images.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Initial learning rate to use.")
    parser.add_argument("--max_train_steps", type=int, default=500, help="Total number of training steps to perform.")
    parser.add_argument("--log_every_steps", type=int, default=50, help="Log images every N steps.")
    parser.add_argument("--num_dataloader_workers", type=int, default=4, help="Number of workers for the dataloader.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb", project_config=project_config)

    # Setup wandb
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth-fibrosis-controlnet", config=vars(args))

    # --- Load Models ---
    model_id = "runwayml/stable-diffusion-v1-5"
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_path)
    controlnet = ControlNetModel.from_pretrained(args.pretrained_controlnet_path)

    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    unet = PeftModel.from_pretrained(unet, args.pretrained_model_path)

    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Add new token to tokenizer
    num_added_tokens = tokenizer.add_tokens("<fibrosis-texture>")
    placeholder_token = "<fibrosis-texture>"
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[tokenizer.convert_tokens_to_ids("lung")].clone() # Initialize with "lung" token
    
    # Freeze VAE and all of text_encoder except the new token embedding
    vae.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    params_to_train = itertools.chain(unet.parameters(), text_encoder.get_input_embeddings().parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=args.learning_rate)

    # --- Dataset and Dataloader ---
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        manifest_path=os.path.join(args.data_dir, "manifest_controlnet.csv"),
        data_dir=args.data_dir,
        size=args.resolution,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_dataloader_workers)

    # Prepare everything with our `accelerator`.
    unet, controlnet, text_encoder, optimizer, train_dataloader = accelerator.prepare(
        unet, controlnet, text_encoder, optimizer, train_dataloader
    )
    vae.to(accelerator.device)

    # --- Training ---
    global_step = 0
    
    # Use a single progress bar for the total training steps
    progress_bar = tqdm(
        range(args.max_train_steps), 
        disable=not accelerator.is_local_main_process,
        desc="Overall Training Progress"
    )

    # Calculate epochs needed, but don't use it for the main progress bar
    num_epochs = math.ceil(args.max_train_steps / len(train_dataloader))

    for epoch in range(num_epochs):
        unet.train()
        text_encoder.train()
        controlnet.train()

        for step, batch in enumerate(train_dataloader):

            # If we've already reached the max steps, break out of the inner loop
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(unet):
                optimizer.zero_grad()

                latents = vae.encode(batch["instance_images"].to(accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["instance_prompt_ids"].squeeze(1).to(accelerator.device))[0]
                
                # ControlNet
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=batch["instance_masks"].to(accelerator.device), return_dict=False
                )

                # UNet
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
            
            # Update the main progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            accelerator.log({"train_loss": loss.item()}, step=global_step)
            global_step += 1

            if global_step % args.log_every_steps == 0:
                # Synchronize all processes before starting the evaluation phase
                accelerator.wait_for_everyone()

                # Set all models to eval mode on all GPUs
                unet.eval()
                text_encoder.eval()
                controlnet.eval()
                
                # The main process handles the image generation and logging
                if accelerator.is_main_process:
                    accelerator.print(f"\nStep {global_step}: Generating validation samples...")
                    with torch.no_grad():
                        num_samples = 2
                        sample_masks = batch["instance_masks"][0:num_samples].to(accelerator.device)
                        
                        prompts = [args.instance_prompt, args.class_prompt]
                        
                        text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
                        text_embeddings = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]
                        
                        gen_latents = torch.randn(num_samples, unet.config.in_channels, 32, 32, device=accelerator.device)

                        # Add a progress bar for the sampling loop
                        for t in tqdm(noise_scheduler.timesteps, 
                                      desc="Generating Samples", 
                                      disable=not accelerator.is_local_main_process):
                            down, mid = controlnet(gen_latents, t, encoder_hidden_states=text_embeddings, controlnet_cond=sample_masks, return_dict=False)
                            noise_pred = unet(gen_latents, t, encoder_hidden_states=text_embeddings, down_block_additional_residuals=down, mid_block_additional_residual=mid).sample
                            gen_latents = noise_scheduler.step(noise_pred, t, gen_latents).prev_sample
                            
                        images = vae.decode(gen_latents / vae.config.scaling_factor).sample
                        images = (images / 2 + 0.5).clamp(0, 1)

                        # Log to WandB
                        accelerator.get_tracker("wandb").log({
                            "samples": [
                                wandb.Image(images[0], caption=f"Step {global_step}: Instance Prompt"),
                                wandb.Image(images[1], caption=f"Step {global_step}: Class Prompt"),
                                wandb.Image(sample_masks[0], caption=f"Step {global_step}: Mask 1"),
                                wandb.Image(sample_masks[1], caption=f"Step {global_step}: Mask 2")
                            ]
                        })
                    accelerator.print("Finished generating samples. Resuming training...")

                # Wait for the main process to finish before anyone continues
                accelerator.wait_for_everyone()

                # Set all models back to train mode on all GPUs
                unet.train()
                text_encoder.train()
                controlnet.train()
                
        # If we've already reached the max steps, break out of the outer loop as well
        if global_step >= args.max_train_steps:
            break
            
    progress_bar.close() # Close the main progress bar after the loop

# --- Save final model ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_path, exist_ok=True)
        
        # Wrap saving process in a tqdm bar for user feedback
        with tqdm(total=3, desc="Saving Final Model") as save_bar:
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(os.path.join(save_path, "unet"))
            save_bar.update(1)

            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            unwrapped_text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))
            save_bar.update(1)

            tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
            save_bar.update(1)

        print(f"✅ Final DreamBooth model saved to {save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    main()