# scripts/create_hdf5.py (Final Version with Truncation)
import os
import h5py
import pandas as pd
from tqdm import tqdm
import cv2
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

def process_image_chunk(task_data):
    """
    Worker function: Reads a chunk of image files from disk, processes them,
    and returns them along with their starting index.
    """
    start_index, image_paths_chunk, image_size = task_data
    processed_images = []
    for img_path in image_paths_chunk:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros(image_size, dtype=np.uint8)
            if img.shape != image_size:
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            processed_images.append(img)
        except Exception as e:
            print(f"Worker failed on {img_path}: {e}", flush=True)
            processed_images.append(np.zeros(image_size, dtype=np.uint8))
    return start_index, processed_images

def create_hdf5_main(data_dir, manifest_file, output_filename, image_size=(256, 256)):
    """
    Main function to orchestrate parallel image processing and serial HDF5 writing.
    """
    manifest_path = os.path.join(data_dir, manifest_file)
    images_dir = os.path.join(data_dir, "images")
    output_path = os.path.join(data_dir, output_filename)

    print(f"Reading original manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    
    # --- FIX: Round down to the nearest full chunk size ---
    chunk_size = 512
    num_images_original = len(manifest)
    num_images_to_process = (num_images_original // chunk_size) * chunk_size
    
    print(f"Original images: {num_images_original}. Processing a truncated set of {num_images_to_process} to ensure full chunks.")

    # Create and save a new, truncated manifest
    manifest_truncated = manifest.iloc[:num_images_to_process].copy()
    final_manifest_path = os.path.join(data_dir, "manifest_final.csv")
    manifest_truncated.to_csv(final_manifest_path, index=False)
    print(f"Saved final manifest for {num_images_to_process} images to: {final_manifest_path}")

    image_paths = [os.path.join(images_dir, fname) for fname in manifest_truncated['original_file']]

    # 1. Create the HDF5 file with the exact final size
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset(
            'images',
            (num_images_to_process, image_size[0], image_size[1]),
            dtype='uint8',
            chunks=(128, image_size[0], image_size[1]),
            compression="gzip"
        )
    print(f"Initialized HDF5 file at: {output_path}")

    # 2. Prepare for multiprocessing
    num_workers = max(1, cpu_count() - 2)
    tasks = [
        (i, image_paths[i:i + chunk_size], image_size)
        for i in range(0, num_images_to_process, chunk_size)
    ]

    print(f"Starting parallel processing with {num_workers} CPU cores...")
    
    # 3. Create a pool of workers and write results
    with h5py.File(output_path, 'a') as hf:
        dset = hf['images']
        with Pool(processes=num_workers) as pool:
            with tqdm(total=num_images_to_process) as pbar:
                for start_index, image_data_chunk in pool.imap_unordered(process_image_chunk, tasks):
                    end_index = start_index + len(image_data_chunk)
                    dset[start_index:end_index, :, :] = image_data_chunk
                    pbar.update(len(image_data_chunk))

    print(f"\n🎉 HDF5 dataset creation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/hot/Yi-Kuan/Fibrosis/Akul/sd_data")
    parser.add_argument("--manifest_file", type=str, default="manifest.csv")
    parser.add_argument("--output_filename", type=str, default="images.h5")
    args = parser.parse_args()
    
    create_hdf5_main(args.data_dir, args.manifest_file, args.output_filename)