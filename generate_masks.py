# generate_masks.py (Version 6 - The Definitive & Safe Pipeline)
import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, disk

from utils import readmat

def definitive_lung_segmentation(image_np):
    """
    Performs a highly robust, multi-step lung segmentation designed to prevent
    failures on difficult or edge-case CT slices.
    """
    # Step 1: Create a mask of the patient's body.
    # This is a crucial first step to isolate the area of interest and
    # remove all air surrounding the patient.
    body_mask = np.zeros_like(image_np, dtype=np.uint8)
    # Start with a generous threshold to get all tissue.
    initial_body = image_np > 0.25
    labeled_body, num_labels = label(initial_body, return_num=True)
    if num_labels > 0:
        # Find the largest connected component, which is the body.
        region_areas = np.bincount(labeled_body.flat)[1:]
        largest_label = np.argmax(region_areas) + 1
        body_mask[labeled_body == largest_label] = 1
    else: # If body isn't found, we can't proceed.
        return body_mask

    # Step 2: Create the initial lung mask *within* the body mask.
    # We use a more generous percentile now because we are in a confined space.
    # This helps capture fibrotic tissue.
    threshold = np.percentile(image_np[body_mask==1], 20)
    lung_mask = (image_np < threshold) & (body_mask == 1)

    # Step 3: Clean the lung mask and separate the two lung fields.
    # Closing fills small holes and gaps inside the lungs.
    lung_mask = binary_closing(lung_mask, disk(3))

    # Erosion severs the connection via the trachea. This is a critical step.
    # We use a slightly less aggressive erosion to protect small lung sections.
    eroded_mask = binary_erosion(lung_mask, disk(1))

    # Step 4: Label the remaining regions and keep the two largest.
    labeled_lungs, num_lung_labels = label(eroded_mask, return_num=True)
    if num_lung_labels < 2:
        # If we can't find two separate lungs (e.g., top/bottom slice),
        # we can fall back to the pre-erosion mask. This prevents blackouts.
        final_mask = binary_fill_holes(lung_mask)
        return (final_mask * 255).astype(np.uint8)

    areas = np.bincount(labeled_lungs.flat)[1:]
    num_to_keep = min(2, len(areas))
    largest_labels = np.argsort(areas)[-num_to_keep:] + 1
    
    # Create the final mask by keeping only these regions.
    final_mask = np.isin(labeled_lungs, largest_labels)

    # Step 5: Restore lung size and fill holes.
    # Dilate to counteract the erosion and then fill for a solid mask.
    final_mask = binary_dilation(final_mask, disk(3))
    final_mask = binary_fill_holes(final_mask)

    return (final_mask * 255).astype(np.uint8)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = [p for p in glob(os.path.join(args.img_dir, "*.mat")) if "contour" not in p.lower()]

    if not image_paths:
        print(f"Error: No .mat files found in {args.img_dir}. Please check the directory.")
        return
        
    print(f"Found {len(image_paths)} CT scan images to process.")

    for img_path in tqdm(image_paths, desc="Generating Definitive Masks"):
        try:
            image_tensor = readmat(img_path)
            image_np = image_tensor.squeeze().cpu().numpy()
            
            lung_mask_np = definitive_lung_segmentation(image_np)

            # Only save if the mask is substantial.
            if np.sum(lung_mask_np) < 1000:
                continue

            mask_image = Image.fromarray(lung_mask_np, mode='L')
            base_name = os.path.basename(img_path)
            output_name = base_name.replace(".mat", "_mask.png")
            output_path = os.path.join(args.output_dir, output_name)
            mask_image.save(output_path)
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")

    print(f"\nMask generation complete. Masks saved in: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Definitive Lung Mask Generation Script")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing the original .mat CT scan files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the new PNG masks.")
    args = parser.parse_args()
    main(args)