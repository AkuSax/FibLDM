import os
import argparse
import pandas as pd
from glob import glob

def main(args):
    image_paths = [os.path.basename(p) for p in glob(os.path.join(args.img_dir, '*.mat')) if 'contour' not in p.lower()]
    rows = []
    for img_name in image_paths:
        mask_name = img_name.replace('.mat', '_mask.png')
        mask_path = os.path.join(args.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_name}, expected at {mask_path}")
        rows.append({'image_path': img_name, 'mask_path': mask_name})
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"CSV saved to {args.output_csv} with {len(df)} entries.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate label CSV mapping images to masks.')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing .mat image files.')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing generated PNG masks.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output CSV file.')
    args = parser.parse_args()
    main(args)
