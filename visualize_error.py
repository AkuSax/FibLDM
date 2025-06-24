import torch
from torchvision.utils import save_image
import argparse
import os
from tqdm import tqdm

# Assuming these imports are correct for your project structure
from dataset import ContourDataset
from autoencoder import VAE

def generate_error_map(args):
    """
    Loads a trained VAE, generates a reconstruction, and saves a side-by-side
    comparison including a visualized error map.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    print("Loading trained VAE model...")
    model = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
    try:
        model.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
    except Exception as e:
        print(f"Error loading state dictionary: {e}")
        return
    model.eval()
    print("Model loaded successfully.")

    # --- Load Dataset ---
    torch.manual_seed(42)
    print("Loading dataset...")
    full_dataset = ContourDataset(label_file=args.label_file, img_dir=args.data_dir)
    
    # Let's pick a specific image to analyze, e.g., the one at the specified index
    if args.image_index >= len(full_dataset):
        print(f"Error: --image_index {args.image_index} is out of bounds for dataset of size {len(full_dataset)}")
        return
        
    original_image, _ = full_dataset[args.image_index]
    original_image = original_image.unsqueeze(0).to(device) # Add batch dimension

    print(f"Generating comparison for image index: {args.image_index}")

    # --- Generate Reconstruction and Error Map ---
    with torch.no_grad():
        recon_image, _, _ = model(original_image)

        # Calculate the absolute difference between the original and reconstruction
        # The VAE outputs are in [-1, 1], so the error will be in [0, 2]
        error_map = torch.abs(original_image - recon_image)
        
        # Amplify the error for better visualization. A value of 0 should remain 0 (black).
        # We will scale it so an error of ~0.2 becomes white.
        # This will make even small errors highly visible.
        error_map_amplified = torch.clamp(error_map * 5.0, 0, 1)

    # --- Save Comparison Image ---
    # Combine original, reconstruction, and error map into one grid
    # We need to normalize all images to [0, 1] for saving
    original_display = (original_image + 1) / 2
    recon_display = (recon_image + 1) / 2
    
    # The error map is already in a good range, just ensure it has a channel dimension
    if error_map_amplified.dim() == 3:
        error_map_amplified = error_map_amplified.unsqueeze(1) # Add channel dim if missing

    comparison_grid = torch.cat([original_display, recon_display, error_map_amplified], dim=0)
    
    output_filename = f"error_comparison_idx_{args.image_index}.png"
    save_image(comparison_grid, output_filename, nrow=3)
    
    print(f"\nSuccessfully saved comparison image to: {output_filename}")
    print("Image layout: [Original] - [Reconstruction] - [Amplified Error Map]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VAE Reconstruction Error Visualization Script")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Path to the trained VAE checkpoint.")
    parser.add_argument("--label_file", type=str, required=True, help="Path to the data CSV file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the image directory.")
    parser.add_argument("--latent_dim", type=int, required=True, help="Latent dimension of the VAE model.")
    parser.add_argument("--image_index", type=int, default=100, help="Index of the image from the dataset to compare.")
    args = parser.parse_args()
    generate_error_map(args)