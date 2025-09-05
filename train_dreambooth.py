import argparse
import itertools
import math
import os
from pathlib import Path

import cv2
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import (AutoencoderKL, ControlNetModel, DDPMScheduler,
                       UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import wandb

logger = get_logger(__name__)

class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, instance_prompt, tokenizer, manifest_path, data_dir, size=512):
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance images root {self.instance_data_root} doesn't exist.")
        self.instance_images_path = list(self.instance_data_root.iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images
        self.size = size
        self.tokenizer = tokenizer
        manifest = pd.read_csv(manifest_path)
        self.mask_map = {row['original_file']: row['mask_file'] for _, row in manifest.iterrows()}
        self.masks_dir = os.path.join(data_dir, 'masks')
        self.image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.mask_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((size, size), antialias=True)])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        img_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        instance_image = cv2.resize(instance_image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        example["instance_images"] = self.image_transforms(instance_image).repeat(3, 1, 1)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt, truncation=True, padding="max_length",
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.squeeze()
        img_filename = os.path.basename(img_path)
        if img_filename in self.mask_map:
            mask_path = os.path.join(self.masks_dir, self.mask_map[img_filename])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            example["instance_masks"] = self.mask_transform(mask_img)
        else:
            example["instance_masks"] = torch.zeros(1, self.size, self.size)
        return example

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--pretrained_vae_path", type=str, required=True)
    parser.add_argument("--pretrained_controlnet_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dreambooth-model")
    parser.add_argument("--instance_prompt", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--log_every_steps", type=int, default=100)
    parser.add_argument("--num_dataloader_workers", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb", project_config=project_config)

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth-fibrosis-controlnet", config=vars(args))

    model_id = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_path)
    controlnet = ControlNetModel.from_pretrained(args.pretrained_controlnet_path)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    unet = PeftModel.from_pretrained(unet, args.pretrained_model_path)


    tokenizer.add_tokens("<fibrosis-texture>")
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    vae.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    params_to_train = itertools.chain(unet.parameters(), text_encoder.get_input_embeddings().parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=args.learning_rate)

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir, instance_prompt=args.instance_prompt,
        tokenizer=tokenizer, manifest_path=os.path.join(args.data_dir, "manifest_controlnet.csv"),
        data_dir=args.data_dir, size=args.resolution
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_dataloader_workers, pin_memory=True
    )

    unet, controlnet, text_encoder, optimizer, train_dataloader = accelerator.prepare(
        unet, controlnet, text_encoder, optimizer, train_dataloader
    )
    vae.to(accelerator.device)

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(math.ceil(args.max_train_steps / len(train_dataloader))):
        unet.train()
        controlnet.train()
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps: break
            
            with accelerator.accumulate(unet):
                optimizer.zero_grad()
                latents = vae.encode(batch["instance_images"].to(accelerator.device)).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["instance_prompt_ids"].to(accelerator.device))[0]
                
                down, mid = controlnet(noisy_latents, timesteps, encoder_hidden_states, controlnet_cond=batch["instance_masks"].to(accelerator.device), return_dict=False)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, down_block_additional_residuals=down, mid_block_additional_residual=mid).sample
                
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_train, 1.0)
                optimizer.step()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1

            if global_step % args.log_every_steps == 0:
                accelerator.log({"train_loss": loss.item()}, step=global_step)

                # --- FINAL, ROBUST VALIDATION BLOCK ---
                # 1. Set all models to eval mode on ALL processes
                unet.eval()
                controlnet.eval()
                text_encoder.eval()
                
                # 2. Synchronize before the main process begins its work
                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    accelerator.print(f"\nStep {global_step}: Generating validation samples...")
                    with torch.no_grad():
                        num_samples = min(2, batch["instance_masks"].shape[0])
                        sample_masks = batch["instance_masks"][:num_samples].to(accelerator.device)
                        gen_latents = torch.randn(num_samples, unet.config.in_channels, args.resolution // 8, args.resolution // 8, device=accelerator.device)
                        
                        for t in noise_scheduler.timesteps:
                            # Use the same prompt embeddings for all samples
                            down, mid = controlnet(gen_latents, t, encoder_hidden_states[:num_samples], controlnet_cond=sample_masks, return_dict=False)
                            noise_pred = unet(gen_latents, t, encoder_hidden_states[:num_samples], down_block_additional_residuals=down, mid_block_additional_residual=mid).sample
                            gen_latents = noise_scheduler.step(noise_pred, t, gen_latents).prev_sample
                        
                        images = vae.decode(gen_latents / vae.config.scaling_factor).sample
                        images = (images / 2 + 0.5).clamp(0, 1)
                        accelerator.get_tracker("wandb").log({"samples": [wandb.Image(img) for img in images]})
                    
                    # Force the GPU to finish all generation work before proceeding
                    torch.cuda.synchronize()
                    accelerator.print("Finished generating samples.")

                # 3. Synchronize again after the main process is finished
                accelerator.wait_for_everyone()
                accelerator.print("Resuming training...")


        if global_step >= args.max_train_steps: break
    
    progress_bar.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(save_path, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(os.path.join(save_path, "unet"))
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        unwrapped_text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))
        tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
        print(f"✅ Final model saved to {save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    main()