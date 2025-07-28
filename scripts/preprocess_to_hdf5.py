# scripts/preprocess_to_hdf5.py
import os
import h5py
import torch
from tqdm import tqdm
import pandas as pd
import argparse

def main(args):
    # --- Load the full dataset manifest ---
    try:
        full_manifest = pd.read_csv(args.full_manifest_path)
        # --- FIX: Create a new column with just the base filenames for matching ---
        full_manifest['basename'] = full_manifest['original_file'].apply(os.path.basename)
    except FileNotFoundError:
        print(f"Error: Full manifest not found at {args.full_manifest_path}")
        return

    # --- Load the desired subset of filenames ---
    try:
        subset_manifest = pd.read_csv(args.subset_csv_path)
        # Use the 'original_file' column to get the filenames
        subset_filenames = set(subset_manifest['original_file'])
        print(f"Found {len(subset_filenames)} unique filenames in the subset CSV.")
    except FileNotFoundError:
        print(f"Error: Subset CSV not found at {args.subset_csv_path}")
        return

    # --- Find matching files by comparing base filenames ---
    files_to_process = full_manifest[full_manifest['basename'].isin(subset_filenames)]
    num_samples = len(files_to_process)
    
    if num_samples == 0:
        print("Error: No matching files found. Please ensure filenames in the subset CSV exist in the original dataset.")
        return
        
    print(f"Found {num_samples} matching files to process.")

    # --- Configuration ---
    base_dir = os.path.dirname(args.full_manifest_path)
    latent_dir = os.path.join(base_dir, 'latents/')
    contour_dir = os.path.join(base_dir, 'contours/')
    output_h5_file = os.path.join(base_dir, 'latents_dataset_subset.h5')

    # --- Get latent shape ---
    first_file_info = files_to_process.iloc[0]
    first_latent_path = os.path.join(latent_dir, f"{first_file_info['index']}.pt")
    
    try:
        first_latent = torch.load(first_latent_path, weights_only=True)
        if first_latent.ndim == 4:
            first_latent = first_latent.squeeze(0)
        latent_shape = first_latent.shape
    except FileNotFoundError:
        print(f"Error: Could not find latent file {first_latent_path}")
        return

    # --- Create the HDF5 file ---
    with h5py.File(output_h5_file, 'w') as f:
        dset_latents = f.create_dataset('latents', (num_samples,) + latent_shape, dtype='f4')
        dset_contours = f.create_dataset('contours', (num_samples,) + (1, 256, 256), dtype='f4')

        print(f"Processing {num_samples} files and saving to {output_h5_file}...")
        # Use .values to iterate faster
        for i, row in tqdm(enumerate(files_to_process[['index']].values), total=num_samples):
            file_idx = row[0]
            latent = torch.load(os.path.join(latent_dir, f'{file_idx}.pt'), weights_only=True)
            contour = torch.load(os.path.join(contour_dir, f'{file_idx}.pt'), weights_only=True)

            if latent.ndim == 4:
                latent = latent.squeeze(0)
            
            dset_latents[i] = latent.numpy()
            dset_contours[i] = contour.numpy()

    print(f"\nSuccessfully created subset dataset at {output_h5_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess a subset of latents to HDF5.")
    parser.add_argument('--subset_csv_path', type=str, required=True, help='Path to the CSV file defining the subset (e.g., clustered_subset.csv).')
    parser.add_argument('--full_manifest_path', type=str, required=True, help='Path to the manifest.csv of the full encoded dataset.')
    args = parser.parse_args()
    main(args)