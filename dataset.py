import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from utils import readmat
from torch.utils.data.sampler import Sampler
import numpy as np
from torchvision import transforms
import torchvision.transforms.v2 as T
import torch.nn.functional as F
import warnings
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

class NonUniformScaling: 
    """
    Apply non-uniform scaling on either direction (vertical or horizontal).

    Parameters:
    scale_x_range (tuple): Range of horizontal scaling.
    scale_y_range (tuple): Range of vertical scaling.

    Returns:
    torch.Tensor: image after scaling.
    """
    def __init__(self, scale_x_range, scale_y_range):
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range

    def __call__(self, img_tensor):
        # Generate random scaling factors using PyTorch
        scale_x = (self.scale_x_range[1] - self.scale_x_range[0]) * torch.rand(1).item() + self.scale_x_range[0]
        scale_y = (self.scale_y_range[1] - self.scale_y_range[0]) * torch.rand(1).item() + self.scale_y_range[0]

        # Create the 2D affine transformation matrix for scaling
        theta = torch.tensor([
            [scale_x, 0, 0],
            [0, scale_y, 0]
        ], dtype=torch.float).unsqueeze(0)  # Add batch dimension

        # Create the affine grid
        grid = torch.nn.functional.affine_grid(theta, img_tensor.size(), align_corners=True)

        # Apply the affine transformation
        stretched_img_tensor = torch.nn.functional.grid_sample(img_tensor, grid, align_corners=True)

        return stretched_img_tensor

class ContourDataset(Dataset):
    def __init__(self, label_file, img_dir, istransform=True, image_size=256):
        self.img_labels = pd.read_csv(label_file) # list file
        self.img_dir   = img_dir # data folder
        self.istransform = istransform
        self.image_size = image_size
        # A more standard and stable augmentation pipeline for medical images
        tfs = []
        if self.image_size is not None:
            tfs.append(transforms.Resize((self.image_size, self.image_size)))
        tfs.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=5,          # Reduced rotation
                translate=(0.05, 0.05) # Reduced translation
            ),
            transforms.Lambda(lambda x: torch.clamp(x, -1.0, 1.0)),
        ])
        self.transforms = transforms.Compose(tfs)


    def __len__(self): # total data number
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        c_name = self.img_labels.iloc[idx, 1]
        # --- Robust image path resolution ---
        img_path_candidates = [
            os.path.join(self.img_dir, "images", img_name),
            os.path.join(self.img_dir, img_name)
        ]
        img_path = next((p for p in img_path_candidates if os.path.exists(p)), None)
        if img_path is None:
            raise FileNotFoundError(f"Image file not found in any expected location: {img_path_candidates}")
        # --- Robust contour path resolution ---
        c_path_candidates = [
            os.path.join(self.img_dir, "contours", c_name),
            os.path.join(self.img_dir, c_name)
        ]
        c_path = next((p for p in c_path_candidates if os.path.exists(p)), None)
        if c_path is None:
            raise FileNotFoundError(f"Contour file not found in any expected location: {c_path_candidates}")
        if img_path.lower().endswith('.png'):
            # Load PNG image as tensor, ensure it's float and normalized to [-1, 1]
            volume = read_image(img_path).float() / 255.0
            if volume.ndim == 3 and volume.shape[0] == 1:
                pass  # already (1, H, W)
            elif volume.ndim == 3 and volume.shape[0] > 1:
                volume = volume[0:1]  # take first channel if RGB
            else:
                volume = volume.unsqueeze(0)
            volume = volume * 2.0 - 1.0  # scale to [-1, 1]
        else:
            volume = readmat(img_path)
        if c_path.lower().endswith('.png'):
            # Load PNG mask as tensor, ensure it's float and binarized
            contour = read_image(c_path).float() / 255.0
            if contour.ndim == 3 and contour.shape[0] == 1:
                pass  # already (1, H, W)
            elif contour.ndim == 3 and contour.shape[0] > 1:
                contour = contour[0:1]  # take first channel if RGB
            else:
                contour = contour.unsqueeze(0)
            contour = (contour > 0.2).float()
        else:
            contour = readmat(c_path)
            contour = (contour > 0.2).float()
        if self.istransform:
            rng_state = torch.random.get_rng_state()
            volume = volume.unsqueeze(0)
            contour = contour.unsqueeze(0)
            volume = self.transforms(volume)
            torch.random.set_rng_state(rng_state)
            contour = self.transforms(contour)
            volume = volume.squeeze(0)
            contour = contour.squeeze(0)
            volume = (volume - volume.min()) / (volume.max() - volume.min())
            volume = volume * 2.0 - 1.0
            volume = volume.float()
        assert volume.shape[0] == 1, f"Volume channels must be 1, got {volume.shape[0]}"
        assert contour.shape[0] == 1, f"Contour channels must be 1, got {contour.shape[0]}"
        assert volume.min() >= -1.0 and volume.max() <= 1.0, "Volume should be in [-1, 1] range."
        assert contour.min() >= 0.0 and contour.max() <= 1.0, "Contour should be in [0, 1] range (binary)."
        # Normalize contour to [-1, 1] regardless of augmentation
        contour = contour * 2.0 - 1.0
        return volume, contour

class ControlNetDataset(Dataset):
    """
    Dataset for ControlNet training that loads CT images and their contour conditioning signals.
    This dataset is designed to work with the ControlNet architecture for precise contour control.
    """
    def __init__(self, label_file, img_dir, img_size=256, latent_size=16, istransform=True):
        self.img_labels = pd.read_csv(label_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.latent_size = latent_size
        self.istransform = istransform
        
        # Augmentation pipeline for ControlNet training
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05)
            ),
            transforms.Lambda(lambda x: torch.clamp(x, -1.0, 1.0)),
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        c_name = self.img_labels.iloc[idx, 1]
        # --- Robust image path resolution ---
        img_path_candidates = [
            os.path.join(self.img_dir, "images", img_name),
            os.path.join(self.img_dir, img_name)
        ]
        img_path = next((p for p in img_path_candidates if os.path.exists(p)), None)
        if img_path is None:
            raise FileNotFoundError(f"Image file not found in any expected location: {img_path_candidates}")
        # --- Robust contour path resolution ---
        c_path_candidates = [
            os.path.join(self.img_dir, "contours", c_name),
            os.path.join(self.img_dir, c_name)
        ]
        c_path = next((p for p in c_path_candidates if os.path.exists(p)), None)
        if c_path is None:
            raise FileNotFoundError(f"Contour file not found in any expected location: {c_path_candidates}")
        # Load CT image
        if img_path.lower().endswith('.png'):
            volume = read_image(img_path).float() / 255.0
            if volume.ndim == 3 and volume.shape[0] == 1:
                pass
            elif volume.ndim == 3 and volume.shape[0] > 1:
                volume = volume[0:1]
            else:
                volume = volume.unsqueeze(0)
            volume = volume * 2.0 - 1.0
        else:
            volume = readmat(img_path)
        # Load contour conditioning signal
        if c_path.lower().endswith('.png'):
            contour = read_image(c_path).float() / 255.0
            if contour.ndim == 3 and contour.shape[0] == 1:
                pass
            elif contour.ndim == 3 and contour.shape[0] > 1:
                contour = contour[0:1]
            else:
                contour = contour.unsqueeze(0)
            contour = (contour > 0.2).float()
        else:
            contour = readmat(c_path)
            contour = (contour > 0.2).float()
        # --- DOWNSAMPLE CONTOUR TO LATENT SIZE BEFORE TRANSFORMS ---
        if self.latent_size is not None:
            contour = F.interpolate(
                contour.unsqueeze(0),
                size=(self.latent_size, self.latent_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        # ----------------------------------------------------------
        # Apply synchronized transformations
        if self.istransform:
            rng_state = torch.random.get_rng_state()
            volume = volume.unsqueeze(0)
            contour = contour.unsqueeze(0)
            volume = self.transforms(volume)
            torch.random.set_rng_state(rng_state)
            contour = self.transforms(contour)
            volume = volume.squeeze(0)
            contour = contour.squeeze(0)
            volume = (volume - volume.min()) / (volume.max() - volume.min())
            volume = volume * 2.0 - 1.0
            volume = volume.float()
        # Normalize contour to [-1, 1] consistently
        contour = contour * 2.0 - 1.0
        return {
            "image": volume,  # CT image for VAE encoding
            "conditioning_image": contour  # Contour for ControlNet conditioning
        }

# Other classes in dataset.py (ContourDataset, etc.) remain the same

class LatentDataset(Dataset):
    """
    A dataset that loads pre-computed latent representations and their
    corresponding contour maps from individual files.
    """
    def __init__(self, data_dir, downsample_contour=False, latent_size=16):
        self.latent_dir = os.path.join(data_dir, "latents")
        self.contour_dir = os.path.join(data_dir, "contours")
        self.manifest_path = os.path.join(data_dir, "manifest.csv")
        
        if not all(os.path.exists(p) for p in [self.latent_dir, self.contour_dir, self.manifest_path]):
            raise FileNotFoundError(f"One or more required directories/files not found in {data_dir}. Please run the `encode_dataset.py` script first.")
            
        self.manifest = pd.read_csv(self.manifest_path)
        self.num_files = len(self.manifest)
        self.downsample_contour = downsample_contour
        self.latent_size = latent_size

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        latent_path = os.path.join(self.latent_dir, f"{idx}.pt")
        contour_path = os.path.join(self.contour_dir, f"{idx}.pt")
        
        latent = torch.load(latent_path)
        contour = torch.load(contour_path)
        
        assert latent.ndim == 3 and latent.shape[1] == self.latent_size and latent.shape[2] == self.latent_size, \
            f"Expected latent shape (C, {self.latent_size}, {self.latent_size}), got {latent.shape}"

        if self.downsample_contour:
            contour = F.interpolate(contour.unsqueeze(0), size=(self.latent_size, self.latent_size), mode='nearest')
            contour = contour.squeeze(0)
            
        return latent, contour
    
class ImageFileDataset(Dataset):
    """A simple dataset to load images from a CSV file of filenames."""
    def __init__(self, label_file, img_dir, istransform=True, image_size=256):
        self.labels = pd.read_csv(label_file)
        self.img_dir = img_dir
        self.istransform = istransform
        self.image_size = image_size
        
        # Expects a single 'image' column with filenames
        if 'image' not in self.labels.columns:
            raise ValueError("CSV file must have a column named 'image'")

        if self.istransform:
            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
                T.Normalize([0.5], [0.5])
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, "images", img_name)
        
        if not os.path.exists(img_path):
             # Fallback for paths that might already be relative
            img_path_alt = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path_alt):
                raise FileNotFoundError(f"Image not found at {img_path} or {img_path_alt}")
            img_path = img_path_alt

        image = Image.open(img_path).convert("RGB")
        
        if self.istransform:
            image = self.transform(image)
            
        return image
    