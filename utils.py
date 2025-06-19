from scipy.io import loadmat
import torch
import mat73

def readmat(filename, device=None):
    m = loadmat(filename)
    L = list(m.keys())
    # print(L)
    cube = m[L[3]]
    # print(cube.dtype)
    # assert False
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