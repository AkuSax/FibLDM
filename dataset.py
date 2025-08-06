# dataset.py (Corrected for Broadcasting and Warnings)
import os
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional")

class HDF5ImageDataset(Dataset):
    # This class is correct and needs no changes.
    def __init__(self, data_dir, manifest_file='manifest.csv', h5_file='images.h5', image_size=256):
        self.data_dir = data_dir
        self.h5_path = os.path.join(self.data_dir, h5_file)
        self.manifest_path = os.path.join(self.data_dir, manifest_file)
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"HDF5 file not found at {self.h5_path}.")
        self.manifest = pd.read_csv(self.manifest_path)
        self.image_size = image_size
        self.h5_file = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
        ])
    def __len__(self):
        return len(self.manifest)
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        if torch.is_tensor(idx):
            idx = idx.item()
        image = self.h5_file['images'][idx]
        image = self.transform(image).repeat(3, 1, 1)
        image = (image * 2.0) - 1.0
        return image.clamp(-1.0, 1.0)

class LatentDataset(Dataset):
    """
    Loads pre-computed latent vectors and constructs text prompts for conditioning.
    """
    def __init__(self, data_dir, manifest_file='manifest_final.csv'):
        self.data_dir = data_dir
        self.latents_dir = os.path.join(self.data_dir, 'latents')
        manifest_path = os.path.join(self.data_dir, manifest_file)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")
        print(f"--- Loading manifest from: {manifest_path} ---")
        self.manifest = pd.read_csv(manifest_path)
        stats_path = os.path.join(self.data_dir, 'latent_stats.pt')
        if os.path.exists(stats_path):
            # FIX: Load with weights_only=True to suppress the warning
            stats = torch.load(stats_path, weights_only=True)
            # FIX: Use .clone().detach() to suppress the warning
            self.latent_mean = stats["mean"].clone().detach()
            self.latent_std = stats["std"].clone().detach()
        else:
            self.latent_mean = 0.0
            self.latent_std = 1.0
            print("Warning: latent_stats.pt not found.")
    def __len__(self):
        return len(self.manifest)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        item_info = self.manifest.iloc[int(idx)]
        latent_index = item_info['index']
        condition = item_info['condition']
        slice_num = item_info['slice']
        latent_filename = f"{latent_index}.pt"
        latent_path = os.path.join(self.latents_dir, latent_filename)
        try:
            latent = torch.load(latent_path, weights_only=True)
        except FileNotFoundError:
            return torch.zeros(4, 32, 32), "error"
        if latent.ndim == 4:
            latent = latent.squeeze(0)
        
        # --- FIX: Reshape mean and std for correct broadcasting ---
        mean = self.latent_mean.view(4, 1, 1)
        std = self.latent_std.view(4, 1, 1)
        latent = (latent - mean) / std
        
        prompt = f"A transverse lung CT scan of a {condition} lung, slice {slice_num}"
        return latent, prompt