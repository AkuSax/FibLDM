import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.utilities.data import dim_zero_cat
import numpy as np
import logging

class RealismMetrics:
    def __init__(self, device='cuda', sync_on_compute=True):
        """Initialize metrics."""
        self.device = device
        self.sync_on_compute = sync_on_compute
        
        # Initialize FID metric
        self.fid = FrechetInceptionDistance(normalize=True, sync_on_compute=sync_on_compute).to(device)
        
        # Initialize KID metric with proper subset_size
        self.kid = KernelInceptionDistance(subset_size=50, normalize=True, sync_on_compute=sync_on_compute).to(device)
        
        # Initialize LPIPS metric
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', sync_on_compute=sync_on_compute).to(device)
        
        # Initialize SSIM metric
        self.ssim = StructuralSimilarityIndexMeasure(sync_on_compute=sync_on_compute).to(device)
        
    def compute_metrics(self, real_images, fake_images, same_mask_images=None):
        """Compute all metrics between real and fake images."""
        metrics = {}
        
        # Ensure inputs are in [0,1] range
        real_images = torch.clamp(real_images, 0, 1)
        fake_images = torch.clamp(fake_images, 0, 1)
        if same_mask_images is not None:
            same_mask_images = torch.clamp(same_mask_images, 0, 1)
        else:
            same_mask_images = real_images
            
        # Convert to 3 channels for FID/KID
        real_3ch = real_images.repeat(1, 3, 1, 1) if real_images.shape[1] == 1 else real_images
        fake_3ch = fake_images.repeat(1, 3, 1, 1) if fake_images.shape[1] == 1 else fake_images
        
        # Normalize to [-1,1] for LPIPS
        real_lpips = (real_images * 2 - 1).repeat(1, 3, 1, 1) if real_images.shape[1] == 1 else (real_images * 2 - 1)
        fake_lpips = (fake_images * 2 - 1).repeat(1, 3, 1, 1) if fake_images.shape[1] == 1 else (fake_images * 2 - 1)
        same_mask_lpips = (same_mask_images * 2 - 1).repeat(1, 3, 1, 1) if same_mask_images.shape[1] == 1 else (same_mask_images * 2 - 1)
        
        # Ensure batch sizes match
        min_batch = min(real_images.shape[0], fake_images.shape[0])
        real_3ch = real_3ch[:min_batch]
        fake_3ch = fake_3ch[:min_batch]
        real_lpips = real_lpips[:min_batch]
        fake_lpips = fake_lpips[:min_batch]
        same_mask_lpips = same_mask_lpips[:min_batch]
        
        try:
            # Reset FID metric
            self.fid.reset()
            # Update with real and fake images
            self.fid.update(real_3ch, real=True)  # Use True instead of tensor
            self.fid.update(fake_3ch, real=False)  # Use False instead of tensor
            # Compute FID
            metrics['fid'] = self.fid.compute().item()
        except Exception as e:
            logging.warning(f"[metrics.py] FID computation error: {str(e)}")
            metrics['fid'] = float('inf')
            
        try:
            # Reset KID metric with dynamic subset_size
            subset_size = min(50, min_batch // 2)  # Use half of batch size, max 50
            if subset_size < 1:
                raise ValueError("Batch size too small for KID computation")
            self.kid = KernelInceptionDistance(subset_size=subset_size, normalize=True, sync_on_compute=self.sync_on_compute).to(real_3ch.device)
            # Update with real and fake images
            self.kid.update(real_3ch, real=True)  # Use True instead of tensor
            self.kid.update(fake_3ch, real=False)  # Use False instead of tensor
            # Compute KID - returns (mean, std)
            kid_mean, kid_std = self.kid.compute()
            metrics['kid'] = kid_mean.item()  # Store mean KID value
            metrics['kid_std'] = kid_std.item()  # Store KID std if needed
        except Exception as e:
            logging.warning(f"[metrics.py] KID computation error: {str(e)}")
            metrics['kid'] = float('inf')
            metrics['kid_std'] = float('inf')
            
        try:
            # Process LPIPS in smaller batches to avoid memory issues
            batch_size = 8  # Process 8 images at a time
            lpips_values = []
            for i in range(0, min_batch, batch_size):
                end_idx = min(i + batch_size, min_batch)
                batch_lpips = self.lpips(
                    fake_lpips[i:end_idx], 
                    same_mask_lpips[i:end_idx]
                ).item()
                lpips_values.append(batch_lpips)
            metrics['lpips'] = np.mean(lpips_values)
        except Exception as e:
            logging.warning(f"[metrics.py] LPIPS computation error: {str(e)}")
            metrics['lpips'] = float('inf')
            
        try:
            metrics['ssim'] = self.ssim(fake_images[:min_batch], real_images[:min_batch]).item()
        except Exception as e:
            logging.warning(f"[metrics.py] SSIM computation error: {str(e)}")
            metrics['ssim'] = float('inf')
            
        return metrics
    
    def reset(self):
        """Reset all metrics"""
        self.fid.reset()
        self.kid.reset()
        self.lpips.reset()
        self.ssim.reset()

    def fid(self, real_images, fake_images):
        """Compute FID between real and fake images."""
        try:
            # Convert to 3 channels for FID
            real_3ch = real_images.repeat(1, 3, 1, 1) if real_images.shape[1] == 1 else real_images
            fake_3ch = fake_images.repeat(1, 3, 1, 1) if fake_images.shape[1] == 1 else fake_images
            
            # Ensure batch sizes match
            min_batch = min(real_3ch.shape[0], fake_3ch.shape[0])
            real_3ch = real_3ch[:min_batch]
            fake_3ch = fake_3ch[:min_batch]
            
            # Reset metric
            self.fid.reset()
            # Update with real and fake images
            self.fid.update(real_3ch, real=True)
            self.fid.update(fake_3ch, real=False)
            # Compute FID
            return self.fid.compute()
        except Exception as e:
            logging.warning(f"[metrics.py] FID computation error: {str(e)}")
            return torch.tensor(float('inf'), device=real_images.device)

    def kid(self, real_images, fake_images):
        """Compute KID between real and fake images."""
        try:
            # Convert to 3 channels for KID
            real_3ch = real_images.repeat(1, 3, 1, 1) if real_images.shape[1] == 1 else real_images
            fake_3ch = fake_images.repeat(1, 3, 1, 1) if fake_images.shape[1] == 1 else fake_images
            
            # Ensure batch sizes match
            min_batch = min(real_3ch.shape[0], fake_3ch.shape[0])
            real_3ch = real_3ch[:min_batch]
            fake_3ch = fake_3ch[:min_batch]
            
            # Calculate subset size
            subset_size = min(50, min_batch // 2)  # Use half of batch size, max 50
            if subset_size < 1:
                raise ValueError("Batch size too small for KID computation")
                
            # Create new KID metric with appropriate subset size
            kid_metric = KernelInceptionDistance(subset_size=subset_size, normalize=True, sync_on_compute=self.sync_on_compute).to(real_3ch.device)
            # Update with real and fake images
            kid_metric.update(real_3ch, real=True)
            kid_metric.update(fake_3ch, real=False)
            # Compute KID
            return kid_metric.compute()
        except Exception as e:
            logging.warning(f"[metrics.py] KID computation error: {str(e)}")
            return (torch.tensor(float('inf'), device=real_images.device), 
                   torch.tensor(float('inf'), device=real_images.device)) 