import torch
import os
import argparse
from tqdm import tqdm
import pandas as pd

from dataset import ContourDataset
from autoencoder import VAE

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Models ---
    print("Loading trained VAE model...")
    vae = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint))
    vae.eval()

    # --- Load Original Dataset ---
    # We set istransform=False to get the raw, un-augmented images
    print("Loading original dataset...")
    dataset = ContourDataset(label_file=args.csv_file, img_dir=args.img_dir, istransform=False)
    
    # --- Create Directories for Latent Data ---
    latent_dir = os.path.join(args.output_dir, "latents")
    contour_dir = os.path.join(args.output_dir, "contours")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(contour_dir, exist_ok=True)

    print(f"Encoding dataset and saving to {args.output_dir}...")
    with torch.no_grad():
        # We need to save the mapping from original file to index
        manifest = []
        for i in tqdm(range(len(dataset))):
            image, contour = dataset[i]
            # Add batch dimension and send to device
            image = image.unsqueeze(0).to(device)

            # Encode the image to get the mean of the latent distribution
            mu, _ = vae.encode(image)

            # Save the latent and contour
            latent_path = os.path.join(latent_dir, f"{i}.pt")
            contour_path = os.path.join(contour_dir, f"{i}.pt")
            
            torch.save(mu.cpu(), latent_path)
            torch.save(contour, contour_path)
            
            # Get original file info from the dataset's dataframe
            original_img_path = dataset.img_labels.iloc[i, 0]
            manifest.append({"index": i, "original_file": original_img_path})

    # Save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join(args.output_dir, "manifest.csv"), index=False)
    print("Encoding complete. Manifest saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Encoding Script")
    # Paths
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the training data CSV file (e.g., ./data/small_label.csv).")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the image directory (e.g., ./data/).")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Path to the trained VAE model checkpoint (e.g., vae_run_1/vae_best.pth).")
    parser.add_argument("--output_dir", type=str, default="./data/latents_dataset", help="Directory to save the encoded latents and contours.")
    
    # Model Hyperparameters
    parser.add_argument("--latent_dim", type=int, default=8, help="Dimension of the VAE latent space. Must match the trained VAE.")
    
    args = parser.parse_args()
    main(args) 