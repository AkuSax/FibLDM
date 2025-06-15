import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.utilities.data import dim_zero_cat
import numpy as np

class RealismMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize metrics
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.kid = KernelInceptionDistance(subset_size=50, normalize=True).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        
    def compute_metrics(self, real_images, fake_images, same_mask_images=None):
        """
        Compute FID, KID, LPIPS and SSIM metrics between real and generated images.
        
        Args:
            real_images (torch.Tensor): Real images [B, C, H, W] in range [0,1]
            fake_images (torch.Tensor): Generated images [B, C, H, W] in range [0,1]
            same_mask_images (torch.Tensor, optional): Real images with same mask as generated
                                                      for LPIPS/SSIM comparison
        """
        metrics = {}
        
        # Convert to uint8 for FID/KID
        real_uint8 = (real_images * 255).to(torch.uint8)
        fake_uint8 = (fake_images * 255).to(torch.uint8)

        # Convert grayscale to 3-channel for FID/KID
        if real_uint8.shape[1] == 1:
            real_uint8 = real_uint8.repeat(1, 3, 1, 1)
        if fake_uint8.shape[1] == 1:
            fake_uint8 = fake_uint8.repeat(1, 3, 1, 1)
        
        # FID
        try:
            fid_result = self.fid.compute()
            if isinstance(fid_result, tuple):
                metrics['fid'] = fid_result[0].item()
            else:
                metrics['fid'] = fid_result.item()
        except Exception as e:
            metrics['fid'] = float('nan')
            print(f"[metrics.py] FID computation error: {e}")
        
        # KID: use a local metric with dynamic subset_size
        min_subset_size = min(real_uint8.shape[0], fake_uint8.shape[0], 50)  # 50 is your preferred max
        try:
            kid_metric = KernelInceptionDistance(subset_size=min_subset_size, normalize=True).to(self.device)
            kid_metric.update(real_uint8, real=True)
            kid_metric.update(fake_uint8, real=False)
            kid_result = kid_metric.compute()
            if isinstance(kid_result, tuple):
                metrics['kid'] = kid_result[0].item()
            else:
                metrics['kid'] = kid_result.item()
        except Exception as e:
            metrics['kid'] = float('nan')
            print(f"[metrics.py] KID computation error: {e}")
        
        # LPIPS and SSIM if same_mask_images provided
        if same_mask_images is not None:
            # LPIPS: ensure 3 channels
            if fake_images.shape[1] == 1:
                fake_images_lpips = fake_images.repeat(1, 3, 1, 1)
            else:
                fake_images_lpips = fake_images
            if same_mask_images.shape[1] == 1:
                same_mask_images_lpips = same_mask_images.repeat(1, 3, 1, 1)
            else:
                same_mask_images_lpips = same_mask_images
            metrics['lpips'] = self.lpips(fake_images_lpips, same_mask_images_lpips).item()
            
            # SSIM
            metrics['ssim'] = self.ssim(fake_images, same_mask_images).item()
            
        return metrics
    
    def reset(self):
        """Reset all metrics"""
        self.fid.reset()
        self.kid.reset()
        self.lpips.reset()
        self.ssim.reset() 