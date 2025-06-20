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
    def __init__(self, label_file, img_dir, istransform=True):
        self.img_labels = pd.read_csv(label_file) # list file
        self.img_dir   = img_dir # data folder
        self.istransform = istransform
        
        self.transforms = transforms.Compose(
        [
            transforms.RandomCrop(256, padding=32, fill=0, padding_mode='constant'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                NonUniformScaling(scale_x_range=[0.8,1.2], scale_y_range=[0.8,1.2]),
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.1,0.1),
                    scale=(0.9,1.1),
                    shear=(0.9,1.1)
                ),
            ], p=0.9),
        ]
        )


    def __len__(self): # total data number
        return len(self.img_labels)

    def __getitem__(self, idx):
        # print(self.img_dir)
        # assert False
        img_name = self.img_labels.iloc[idx, 0]
        c_name = self.img_labels.iloc[idx, 1]
        # print(img_name)
        # print(c_name)
        img_path = os.path.join(self.img_dir, img_name)
        c_path = os.path.join(self.img_dir, c_name)
        # print(img_path)
        # print(c_path)
        # assert False
        volume = readmat(img_path)
        contour = readmat(c_path)

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
            # Normalize to [-1, 1] for diffusion model
            volume = volume * 2.0 - 1.0
            volume = volume.float()

            contour = (contour>0.2).float()

        return volume, contour
    