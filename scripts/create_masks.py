# scripts/create_masks_from_manifest.py (Corrected)
import os
import glob
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def build_mask_map(source_dirs, mask_suffixes):
    """Scans source directories to create a fast lookup map for mask files."""
    print("Scanning source directories to build mask file map...")
    mask_map = {}
    for source_dir in source_dirs:
        for suffix in mask_suffixes:
            for path in glob.glob(os.path.join(source_dir, "**", f"*{suffix}"), recursive=True):
                # --- FIX: Correctly generate the key by removing only the extension ---
                basename = os.path.basename(path)
                key = basename.replace(".nii.gz", "")
                mask_map[key] = path
    print(f"Found {len(mask_map)} unique mask volumes.")
    return mask_map

def process_manifest_entry(task_data, mask_map, output_dir, image_size=(256, 256)):
    """Worker function to find, process, and save a single mask slice."""
    manifest_row, manifest_idx = task_data
    image_filename = manifest_row['original_file']
    slice_index = manifest_row['slice']

    mask_key = image_filename.split('_slice_')[0].replace('_image', '_mask').replace('_img', '_mask')
    
    if mask_key in mask_map:
        mask_volume_path = mask_map[mask_key]
        try:
            mask_volume_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_volume_path))
            mask_slice = mask_volume_np[slice_index, :, :]
            
            mask_processed = cv2.normalize(mask_slice, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            mask_resized = cv2.resize(mask_processed, image_size, interpolation=cv2.INTER_NEAREST)

            new_mask_name = image_filename.replace('_image.png', '_mask.png').replace('_img.png', '_mask.png')
            cv2.imwrite(os.path.join(output_dir, "masks", new_mask_name), mask_resized)
            
            return manifest_idx, new_mask_name
        except Exception:
            return manifest_idx, None
    return manifest_idx, None

def main():
    data_dir = "/hot/Yi-Kuan/Fibrosis/Akul/sd_data"
    manifest_path = os.path.join(data_dir, "manifest_final.csv")
    os.makedirs(os.path.join(data_dir, "masks"), exist_ok=True)

    source_dirs = ["/hot/COPDGene-1", "/hot/Hsu-Ting/FibrosisData"]
    mask_suffixes = ["_INSP_mask.nii.gz", "_mask.nii.gz"]
    mask_map = build_mask_map(source_dirs, mask_suffixes)

    manifest = pd.read_csv(manifest_path)
    tasks = [(row, index) for index, row in manifest.iterrows()]
    
    num_workers = max(1, cpu_count() - 2)
    print(f"Processing {len(manifest)} manifest entries with {num_workers} workers...")

    results = [None] * len(manifest)
    with Pool(processes=num_workers) as pool:
        worker_func = partial(process_manifest_entry, mask_map=mask_map, output_dir=data_dir)
        with tqdm(total=len(tasks)) as pbar:
            for manifest_idx, new_mask_name in pool.imap_unordered(worker_func, tasks):
                if new_mask_name:
                    results[manifest_idx] = new_mask_name
                pbar.update(1)

    manifest['mask_file'] = results
    manifest.dropna(subset=['mask_file'], inplace=True)
    
    final_manifest_path = os.path.join(data_dir, "manifest_controlnet.csv")
    manifest.to_csv(final_manifest_path, index=False)
    
    print(f"\n🎉 Mask processing complete. Final manifest for ControlNet saved to {final_manifest_path}")
    print(f"   - Total entries with matching masks: {len(manifest)}")

if __name__ == "__main__":
    main()