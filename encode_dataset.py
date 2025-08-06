# scripts/encode_dataset.py (Final, Robust Version)
import os
import argparse
import torch
import numpy as np
import pandas as pd
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import multiprocessing as mp

from dataset import HDF5ImageDataset as ImageDatasetForVAE

def encode_worker(gpu_id, manifest_indices, args):
    """
    A worker function that runs on a single GPU to encode a subset of the data.
    """
    device = f"cuda:{gpu_id}"
    print(f"[Process on GPU {gpu_id}]: Starting with {len(manifest_indices)} images.")

    vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(device)
    vae.eval()

    full_dataset = ImageDatasetForVAE(
        data_dir=args.data_dir,
        manifest_file='manifest_final.csv',
        image_size=args.image_size
    )
    worker_dataset = Subset(full_dataset, manifest_indices)
    
    dataloader = DataLoader(
        worker_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers_per_gpu
    )

    latents_dir = os.path.join(args.output_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"GPU {gpu_id} Encoding", position=gpu_id)):
            images = batch.to(device)
            latents = vae.encode(images).latent_dist.mean

            start_idx_in_manifest_chunk = i * args.batch_size
            end_idx_in_manifest_chunk = start_idx_in_manifest_chunk + len(latents)
            original_manifest_indices = manifest_indices[start_idx_in_manifest_chunk:end_idx_in_manifest_chunk]

            for j, manifest_index in enumerate(original_manifest_indices):
                original_file_index = full_dataset.manifest.iloc[manifest_index]['index']
                save_path = os.path.join(latents_dir, f"{original_file_index}.pt")
                torch.save(latents[j].cpu(), save_path)
    
    print(f"[Process on GPU {gpu_id}]: Finished encoding.")

def main():
    parser = argparse.ArgumentParser(description="Encode a dataset in parallel using multiple GPUs.")
    parser.add_argument("--vae_model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size PER GPU.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers_per_gpu", type=int, default=16, help="Number of CPU workers for data loading per GPU process.")
    args = parser.parse_args()

    latents_dir = os.path.join(args.output_dir, "latents")
    if os.path.exists(latents_dir) and len(os.listdir(latents_dir)) > 1000:
         print("✅ Latents directory already exists and is populated. Skipping encoding.")
    else:
        # This block will be skipped since your latents are already created.
        pass

    # --- Final Step: Calculate Stats ---
    print("Calculating statistics...")
    all_latents = []
    latent_files = os.listdir(latents_dir)
    for filename in tqdm(latent_files, desc="Loading latents for stats"):
        if not filename.endswith(".pt"):
            continue
        file_path = os.path.join(latents_dir, filename)
        try:
            data = torch.load(file_path, map_location="cpu")
            if isinstance(data, torch.Tensor):
                # --- FIX: Check for and remove the extra dimension ---
                if data.ndim == 4 and data.shape[0] == 1:
                    data = data.squeeze(0)
                all_latents.append(data)
            else:
                print(f"\nWarning: Skipping non-tensor file: {filename}")
        except Exception as e:
            print(f"\nWarning: Could not load file {filename}, skipping. Error: {e}")

    all_latents_tensor = torch.stack(all_latents, dim=0)
    mean = torch.mean(all_latents_tensor, dim=[0, 2, 3])
    std = torch.std(all_latents_tensor, dim=[0, 2, 3])
    
    stats = {"mean": mean, "std": std}
    stats_path = os.path.join(args.output_dir, "latent_stats.pt")
    torch.save(stats, stats_path)

    print(f"\n🎉 Statistics calculation complete.")
    print(f"   - Mean: {stats['mean'].numpy()}")
    print(f"   - Std: {stats['std'].numpy()}")
    print(f"   - Statistics saved to {stats_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()