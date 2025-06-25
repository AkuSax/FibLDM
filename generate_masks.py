# generate_masks.py (Version 2 - Robust)
import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.morphology import binary_closing, binary_opening

from utils import readmat

def segment_lungs(image_np):
    """
    Segments the lung regions from a CT scan using a multi-step,
    data-adaptive approach.
    """
    # Step 1: Create a general mask of the patient's body to exclude air
    # around the patient. We can do this with a simple threshold above the
    # image's minimum value (which is usually the background air).
    body_mask = image_np > (image_np.min() + 0.01)

    # Step 2: Invert the image so that the lungs (low density) become bright
    # and other tissues (high density) become dark.
    inverted_image = image_np.max() - image_np
    
    # Step 3: Within the body mask, find the brightest regions, which now
    # correspond to the lungs and airways.
    # We use a threshold relative to the brightest parts of the inverted image.
    lungs_threshold = np.quantile(inverted_image[body_mask], 0.9) # Find the 90th percentile brightness
    potential_lungs = inverted_image > lungs_threshold

    # Step 4: Clean up this potential lung mask.
    # Closing operation will fill gaps within the lung regions.
    # Opening will remove small, noisy bright spots.
    potential_lungs = binary_closing(potential_lungs, np.ones((5,5)))
    potential_lungs = binary_opening(potential_lungs, np.ones((5,5)))

    # Step 5: Label each disconnected region.
    labeled_image, num_labels = label(potential_lungs, return_num=True)
    
    # If no regions are found, return an empty mask.
    if num_labels == 0:
        return np.zeros_like(image_np, dtype=np.uint8)

    # Step 6: Find the two largest regions by area. These will be the lungs.
    # We calculate the area of each labeled region.
    region_areas = np.bincount(labeled_image.flat)[1:] # Get area of each label > 0
    
    # Get the labels of the two largest regions.
    # We add a check in case there's only one large region found.
    num_regions_to_keep = min(2, len(region_areas))
    if num_regions_to_keep == 0:
        return np.zeros_like(image_np, dtype=np.uint8)
        
    largest_region_labels = np.argsort(region_areas)[-num_regions_to_keep:] + 1
    
    # Create the final lung mask by keeping only the largest regions.
    lung_mask = np.isin(labeled_image, largest_region_labels)

    # Step 7: Fill any holes inside the final lung masks for a solid shape.
    final_mask = binary_fill_holes(lung_mask)

    return (final_mask * 255).astype(np.uint8)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = [p for p in glob(os.path.join(args.img_dir, "*.mat")) if "contour" not in p.lower()]
    
    if not image_paths:
        print(f"Error: No .mat files found in {args.img_dir}. Please check the directory.")
        return
        
    print(f"Found {len(image_paths)} CT scan images to process.")

    for img_path in tqdm(image_paths):
        try:
            image_tensor = readmat(img_path)
            image_np = image_tensor.squeeze().cpu().numpy()

            lung_mask_np = segment_lungs(image_np)

            if np.sum(lung_mask_np) == 0:
                print(f"Warning: Empty mask generated for {os.path.basename(img_path)}. Check image intensity range.")
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
    parser = argparse.ArgumentParser(description="Robust Lung Mask Generation Script")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing the original .mat CT scan files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the new PNG masks.")
    args = parser.parse_args()
    main(args)