from __future__ import annotations
import torch
import torch.nn.functional as F
from torchmetrics.segmentation.hausdorff_distance import HausdorffDistance
_hd = HausdorffDistance(num_classes=1)


# Dice 
def dice_loss(pred: torch.Tensor,
              target: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(pred)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()

# Boundary / Edge-based loss
def boundary_loss(pred, target):
    # Convert to probabilities
    pred_prob = torch.sigmoid(pred)
    
    # Compute gradients using Sobel filters (edge detection)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=pred_prob.dtype,  # Match input precision
                          device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=pred_prob.dtype,  # Match input precision
                          device=pred.device).view(1, 1, 3, 3)
    
    # Compute edges for both prediction and target
    pred_edges_x = F.conv2d(pred_prob, sobel_x, padding=1)
    pred_edges_y = F.conv2d(pred_prob, sobel_y, padding=1)
    pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)
    
    target_edges_x = F.conv2d(target, sobel_x, padding=1)
    target_edges_y = F.conv2d(target, sobel_y, padding=1)
    target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)
    
    # Compute edge loss
    edge_loss = F.mse_loss(pred_edges, target_edges)
    
    return edge_loss

# Focal
def focal_loss(pred: torch.Tensor,
               target: torch.Tensor,
               alpha: float = 0.25,
               gamma: float = 2.0,
               eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(pred).clamp(min=eps, max=1-eps)
    pt = p * target + (1.0 - p) * (1.0 - target)
    w  = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (-w * (1 - pt).pow(gamma) * pt.log()).mean()

# PatchGAN adversarial (generator side) 
def adv_loss_G(disc_pred_fake: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        disc_pred_fake, torch.ones_like(disc_pred_fake)
    )

# registry 
LOSS_REGISTRY = {
    "mse":   torch.nn.MSELoss(reduction="mean"),
    "dice":  dice_loss,
    "boundary": boundary_loss,
    "focal": focal_loss,
    "adv":   adv_loss_G,
}
