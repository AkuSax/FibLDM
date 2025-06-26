from scipy.io import loadmat
import torch
import mat73
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt

def readmat(filename, device=None):
    m = loadmat(filename)
    L = list(m.keys())
    cube = m[L[3]]
    cube_tensor = torch.unsqueeze(torch.tensor(cube),0) # make another dimension for channels
    if device:
        cube_tensor=cube_tensor.to(device, dtype=torch.float)    
    else:
        cube_tensor=cube_tensor.to(dtype=torch.float)
    return cube_tensor

def readmat73(filename, device=None):
    m = mat73.loadmat(filename)
    L = list(m.keys())
    # print(L)
    cube = m[L[0]]
    # print(cube.dtype)
    # assert False
    cube_tensor = torch.unsqueeze(torch.tensor(cube),0) # make another dimension for channels
    if device:
        cube_tensor=cube_tensor.to(device, dtype=torch.float)    
    else:
        cube_tensor=cube_tensor.to(dtype=torch.float)
    return cube_tensor

def loadmat_try(filename):
    try:
        cube_tensor = readmat(filename)
    except:
        cube_tensor = readmat73(filename)
    return cube_tensor

import torchvision
import matplotlib.pyplot as plt
import numpy as np
def save_images(images, contour, sample_size, path, **kwargs):
    print(images.shape)
    N = images.shape[0]
    images_show = images.squeeze(1).detach().cpu()
    contour_show = contour.squeeze(1).detach().cpu()
    # Create a grid for displaying the images
    fig, axes = plt.subplots(sample_size, 2, figsize=(8, 32))

    for i in range(sample_size):
        # Plot left image with a grayscale colormap
        axes[i, 0].imshow(images_show[i], cmap='gray')
        axes[i, 0].axis('off')
        
        # Plot right image with a grayscale colormap
        axes[i, 1].imshow(contour_show[i], cmap='gray')
        axes[i, 1].axis('off')

    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()

def ensure_three_channels(x):
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x

def save_debug_samples(epoch, real_images, real_contours, fake_samples, save_dir="debug_samples"):
    """Saves a grid of real images, real contours, and generated samples."""
    os.makedirs(save_dir, exist_ok=True)
    # Ensure tensors are on CPU and in [0, 1] range for visualization
    real_images = (real_images.clamp(-1, 1) + 1) / 2  # from [-1, 1] to [0, 1]
    fake_samples = fake_samples.clamp(0, 1)  # Already in [0, 1]
    n_samples = min(4, real_images.size(0))
    # Ensure all are 3-channel
    real_images_3c = ensure_three_channels(real_images[:n_samples].cpu())
    real_contours_3c = ensure_three_channels(real_contours[:n_samples].cpu())
    fake_samples_3c = ensure_three_channels(fake_samples[:n_samples].cpu())
    comparison_grid = torch.cat([
        real_images_3c,
        real_contours_3c,
        fake_samples_3c
    ], dim=0)
    grid = vutils.make_grid(comparison_grid, nrow=n_samples, normalize=False)
    plt.imsave(
        os.path.join(save_dir, f"epoch_{epoch:03d}.png"),
        grid.permute(1, 2, 0).numpy()
    )

class EarlyStopper:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        if mode == 'min':
            self.best_score = float('inf')
        elif mode == 'max':
            self.best_score = -float('inf')
        else:
            raise ValueError("mode should be 'min' or 'max'")

    def early_stop(self, score):
        if self.mode == 'min':
            improvement = self.best_score - score
            if improvement > self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        else:  # mode == 'max'
            improvement = score - self.best_score
            if improvement > self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register(model)

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}