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