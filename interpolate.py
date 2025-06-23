import torch
from torchvision.utils import save_image
import argparse
import os

# Make sure these imports match your project structure
from dataset import ContourDataset
from autoencoder import VAE

def run_interpolation(args):
    """
    Loads a trained VAE, interpolates between the latent vectors of two images,
    and saves the resulting decoded images as a grid.
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
        print("This often happens if the model architecture in autoencoder.py does not match the saved checkpoint.")
        return

    model.eval()
    print("Model loaded successfully.")

    # --- Load Dataset ---
    print("Loading dataset to select images...")
    try:
        full_dataset = ContourDataset(label_file=args.label_file, img_dir=args.data_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    if max(args.idx1, args.idx2) >= len(full_dataset):
        print(f"Error: Image indices are out of bounds. Dataset has {len(full_dataset)} images.")
        return

    # Get the two images for interpolation
    img1, _ = full_dataset[args.idx1]
    img2, _ = full_dataset[args.idx2]

    # Add batch dimension and send to device
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    
    print(f"Interpolating between image {args.idx1} and {args.idx2} over {args.steps} steps.")

    # --- Perform Interpolation ---
    with torch.no_grad():
        # Encode the two images to get their latent representations
        mu1, logvar1 = model.encode(img1)
        mu2, logvar2 = model.encode(img2)

        z1 = model.reparameterize(mu1, logvar1)
        z2 = model.reparameterize(mu2, logvar2)

        # Create a list to hold the interpolated images
        interpolated_images = []

        # Generate interpolation weights (alpha) from 0.0 to 1.0
        alphas = torch.linspace(0, 1, args.steps)

        for alpha in alphas:
            # Linearly interpolate in the latent space
            z_inter = z1 * (1 - alpha) + z2 * alpha
            
            # Decode the interpolated latent vector
            img_inter = model.decode(z_inter)
            interpolated_images.append(img_inter)

        # Add the original start and end images for comparison
        interpolated_images.insert(0, img1)
        interpolated_images.append(img2)
        
        # Combine all images into a single tensor
        output_grid = torch.cat(interpolated_images)

    # --- Save Output ---
    save_path = os.path.join(args.save_dir, "interpolation_result.png")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save the grid of images
    save_image(output_grid.cpu(), save_path, nrow=args.steps + 2)
    print(f"Successfully saved interpolation grid to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VAE Latent Space Interpolation Script")
    # Paths
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Path to the trained VAE checkpoint (e.g., ../model_runs/vae_run_2/vae_best.pth).")
    parser.add_argument("--label_file", type=str, required=True, help="Path to the data CSV file (e.g., ./data/label.csv).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the image directory (e.g., ./data/).")
    parser.add_argument("--save_dir", type=str, default="interpolation_samples", help="Directory to save the output image grid.")
    
    # Model & Interpolation Hyperparameters
    parser.add_argument("--latent_dim", type=int, required=True, help="Latent dimension of the VAE model.")
    parser.add_argument("--idx1", type=int, default=0, help="Index of the first image in the dataset.")
    parser.add_argument("--idx2", type=int, default=100, help="Index of the second image in the dataset.")
    parser.add_argument("--steps", type=int, default=10, help="Number of interpolation steps between the two images.")
    
    args = parser.parse_args()
    run_interpolation(args)
