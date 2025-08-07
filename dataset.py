# dataset.py (Corrected for ControlNet Channel Mismatch)
import os
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import warnings
import cv2

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
    # This class is correct and needs no changes.
    def __init__(self, data_dir, manifest_file='manifest_final.csv'):
        self.data_dir = data_dir
        self.latents_dir = os.path.join(self.data_dir, 'latents')
        manifest_path = os.path.join(self.data_dir, manifest_file)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")
        self.manifest = pd.read_csv(manifest_path)
        stats_path = os.path.join(self.data_dir, 'latent_stats.pt')
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, weights_only=True)
            self.latent_mean = stats["mean"].clone().detach()
            self.latent_std = stats["std"].clone().detach()
        else:
            self.latent_mean, self.latent_std = 0.0, 1.0
    def __len__(self):
        return len(self.manifest)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        item_info = self.manifest.iloc[int(idx)]
        latent_index = item_info['index']
        condition = item_info['condition']
        slice_num = item_info['slice']
        latent_path = os.path.join(self.latents_dir, f"{latent_index}.pt")
        try:
            latent = torch.load(latent_path, weights_only=True)
        except FileNotFoundError:
            return torch.zeros(4, 32, 32), "error"
        if latent.ndim == 4: latent = latent.squeeze(0)
        mean, std = self.latent_mean.view(4, 1, 1), self.latent_std.view(4, 1, 1)
        latent = (latent - mean) / std
        prompt = f"A transverse lung CT scan of a lung with {condition}"
        return latent, prompt

class ControlNetLatentDataset(Dataset):
    """
    Loads pre-computed latents, their corresponding masks, and constructs text prompts.
    """
    def __init__(self, data_dir, manifest_file='manifest_controlnet.csv'):
        self.data_dir = data_dir
        self.latents_dir = os.path.join(self.data_dir, 'latents')
        self.masks_dir = os.path.join(self.data_dir, 'masks')
        manifest_path = os.path.join(self.data_dir, manifest_file)

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"ControlNet manifest not found at {manifest_path}")
        
        self.manifest = pd.read_csv(manifest_path)

        stats_path = os.path.join(self.data_dir, 'latent_stats.pt')
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, weights_only=True)
            self.latent_mean = stats["mean"].clone().detach()
            self.latent_std = stats["std"].clone().detach()
        else:
            self.latent_mean, self.latent_std = 0.0, 1.0

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True)
        ])

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
            
        item_info = self.manifest.iloc[int(idx)]
        latent_index = item_info['index']
        mask_filename = item_info['mask_file']
        condition = item_info['condition']
        slice_num = item_info['slice']

        latent_path = os.path.join(self.latents_dir, f"{latent_index}.pt")
        try:
            latent = torch.load(latent_path, weights_only=True)
        except FileNotFoundError:
            return None
        
        if latent.ndim == 4: latent = latent.squeeze(0)
        mean, std = self.latent_mean.view(4, 1, 1), self.latent_std.view(4, 1, 1)
        latent = (latent - mean) / std
        
        mask_path = os.path.join(self.masks_dir, mask_filename)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            return None
        mask_tensor = self.mask_transform(mask_img)

        prompt = f"A transverse lung CT scan of a lung with {condition}"
        
        return latent, mask_tensor, prompt