import os
import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file

def convert_accelerate_checkpoint_to_hf(checkpoint_path, output_path):
    """
    Loads a VAE model from an Accelerate checkpoint and saves it in the
    standard Hugging Face format.
    """
    # 1. Define the model architecture we are loading into.
    #    This must match the model that was trained.
    model = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

    # 2. Construct the path to the saved weights file.
    weights_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(weights_path):
        print(f"Error: model.safetensors not found at {weights_path}")
        return

    # 3. Load the state dictionary (the weights) from the file.
    #    'map_location="cpu"' ensures it loads without using GPU memory.
    state_dict = load_file(weights_path, device="cpu")
    
    # 4. Load these weights into our model architecture.
    model.load_state_dict(state_dict)
    
    # 5. Save the model in the standard Hugging Face format.
    model.save_pretrained(output_path)
    print(f"✅ Model successfully converted and saved to: {output_path}")

if __name__ == "__main__":
    # The path to the directory containing your 'model.safetensors' file.
    input_checkpoint_dir = "../model_runs/vae_run_2/best_model"
    
    # The new directory where the converted, ready-to-use model will be saved.
    converted_output_dir = "../model_runs/vae_run_2/best_model_hf"

    convert_accelerate_checkpoint_to_hf(input_checkpoint_dir, converted_output_dir)