import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torchvision.utils import save_image
import argparse
import os
from tqdm import tqdm

from dataset import ContourDataset # Your existing dataset
from autoencoder import VAE
from utils import EarlyStopper

def vae_loss_function(recon_x, x, mu, logvar, kld_weight):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = (recon_loss + kld_weight * kld_loss) / x.size(0) # Average over batch
    return loss, recon_loss / x.size(0), kld_loss / x.size(0)

def main(args):
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    # --- Data ---
    full_dataset = ContourDataset(label_file=args.label_file, img_dir=args.data_dir)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples.")

    # --- Model ---
    model = VAE(in_channels=1, latent_dim=args.latent_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=10, mode='min')

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]")
        for images, _ in pbar:
            images = images.to(device)

            recon_images, mu, logvar = model(images)
            loss, recon_loss, kld_loss = vae_loss_function(recon_images, images, mu, logvar, args.kld_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon_loss.item():.4f}", kld=f"{kld_loss.item():.4f}")
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]")
        with torch.no_grad():
            for images, _ in pbar_val:
                images = images.to(device)
                recon_images, mu, logvar = model(images)
                loss, recon_loss, kld_loss = vae_loss_function(recon_images, images, mu, logvar, args.kld_weight)
                total_val_loss += loss.item()
                pbar_val.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon_loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        # --- Save Checkpoint and Samples ---
        if avg_val_loss < stopper.best_score:
            print(f"New best validation loss: {avg_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(args.save_dir, "vae_best.pth"))
        
        if stopper.early_stop(avg_val_loss):
            print("Early stopping triggered.")
            break

        # Save some reconstruction samples for visualization
        if (epoch + 1) % args.save_interval == 0:
            with torch.no_grad():
                val_images, _ = next(iter(val_loader))
                val_images = val_images.to(device)
                recon_val, _, _ = model(val_images)
                
                # Combine original and reconstructed images
                comparison = torch.cat([val_images[:8], recon_val[:8]])
                save_image(comparison.cpu(), os.path.join(args.save_dir, f"recon_{epoch+1}.png"), nrow=8)
                print(f"Saved reconstruction sample to {os.path.join(args.save_dir, f'recon_{epoch+1}.png')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VAE Training Script")
    # Paths
    parser.add_argument("--label_file", type=str, required=True, help="Path to the training data CSV file (e.g., ./data/small_label.csv).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the image directory (e.g., ./data/).")
    parser.add_argument("--save_dir", type=str, default="vae_checkpoint", help="Directory to save model and samples.")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--latent_dim", type=int, default=8, help="Dimension of the VAE latent space.")
    parser.add_argument("--kld_weight", type=float, default=1e-4, help="Weight for the KL Divergence term in the loss.")
    
    # Misc
    parser.add_argument("--save_interval", type=int, default=5, help="Epoch interval to save reconstruction samples.")
    
    args = parser.parse_args()
    main(args) 