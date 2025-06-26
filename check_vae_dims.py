import torch
import argparse
from autoencoder import VAE

def check_vae_dimensions(checkpoint_path):
    """Check the latent dimensions of a VAE checkpoint."""
    
    print(f"Loading VAE checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n=== Checkpoint Keys ===")
    for key in checkpoint.keys():
        print(f"  {key}")
    
    print("\n=== VAE Architecture Analysis ===")
    
    # Try to infer latent dimensions from the state dict
    state_dict = checkpoint
    
    # Look for encoder output layer (should be latent_dim * 2 for mu and logvar)
    encoder_keys = [k for k in state_dict.keys() if 'encoder' in k]
    print(f"Encoder layers found: {encoder_keys}")
    
    # Look for the final encoder layer
    final_encoder_key = None
    for key in encoder_keys:
        if 'Conv2d' in str(type(state_dict[key])) or 'conv' in key.lower():
            final_encoder_key = key
            break
    
    if final_encoder_key:
        final_layer = state_dict[final_encoder_key]
        print(f"Final encoder layer shape: {final_layer.shape}")
        
        # The final encoder layer should output latent_dim * 2 channels
        # (mu and logvar for each latent dimension)
        if len(final_layer.shape) == 4:  # Conv2d layer
            out_channels = final_layer.shape[0]
            print(f"Output channels: {out_channels}")
            print(f"Inferred latent_dim: {out_channels // 2}")
        else:
            print("Could not determine latent_dim from final layer shape")
    
    # Also check decoder input layer
    decoder_keys = [k for k in state_dict.keys() if 'decoder' in k]
    print(f"Decoder layers found: {decoder_keys}")
    
    # Look for the first decoder layer
    first_decoder_key = None
    for key in decoder_keys:
        if 'ConvTranspose2d' in str(type(state_dict[key])) or 'conv' in key.lower():
            first_decoder_key = key
            break
    
    if first_decoder_key:
        first_layer = state_dict[first_decoder_key]
        print(f"First decoder layer shape: {first_layer.shape}")
        
        if len(first_layer.shape) == 4:  # ConvTranspose2d layer
            in_channels = first_layer.shape[1]
            print(f"Input channels: {in_channels}")
            print(f"Confirmed latent_dim: {in_channels}")
    
    # Try to create a VAE with different latent dimensions and see which one works
    print("\n=== Testing VAE Initialization ===")
    
    for latent_dim in [4, 8, 16, 32, 64]:
        try:
            vae = VAE(in_channels=1, latent_dim=latent_dim)
            vae.load_state_dict(checkpoint)
            print(f"✓ VAE with latent_dim={latent_dim} loads successfully")
            
            # Test forward pass
            test_input = torch.randn(1, 1, 256, 256)
            with torch.no_grad():
                output, mu, logvar = vae(test_input)
                print(f"  Input shape: {test_input.shape}")
                print(f"  Output shape: {output.shape}")
                print(f"  Mu shape: {mu.shape}")
                print(f"  Logvar shape: {logvar.shape}")
                print(f"  Latent space shape: {mu.shape[1:]} (channels, height, width)")
                
                # Calculate latent size
                latent_size = mu.shape[2]  # height/width (should be same)
                print(f"  Latent size: {latent_size}x{latent_size}")
                print(f"  Total latent dimensions: {latent_dim} channels × {latent_size} × {latent_size} = {latent_dim * latent_size * latent_size}")
                break
                
        except Exception as e:
            print(f"✗ VAE with latent_dim={latent_dim} failed: {str(e)[:100]}...")
    
    print("\n=== Summary ===")
    print(f"Recommended settings for train_controlnet.sh:")
    print(f"  LATENT_DIM={latent_dim}")
    print(f"  LATENT_SIZE={latent_size}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check VAE latent dimensions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VAE checkpoint")
    args = parser.parse_args()
    
    check_vae_dimensions(args.checkpoint) 