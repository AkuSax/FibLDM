import torch
import torch.nn as nn
import torch.nn.functional as F
from unet2d import UNet2DLatent, DoubleConv, Down, Up, OutConv, SinusoidalPositionEmbeddings


class ControlNet(nn.Module):
    """
    ControlNet for latent space diffusion with contour conditioning.
    This creates a trainable copy of the UNet's downsampling path that processes
    the contour conditioning signal and injects it into the main UNet.
    """
    def __init__(self, unet: UNet2DLatent, conditioning_channels: int = 1):
        super().__init__()
        
        self.unet = unet
        self.conditioning_channels = conditioning_channels
        
        # Time embedding (shared with main UNet)
        time_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Contour encoder: small CNN to process the conditioning signal
        contour_dim = 64
        self.contour_encoder = nn.Sequential(
            nn.Conv2d(conditioning_channels, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, contour_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # ControlNet input convolution
        self.control_conv_in = nn.Conv2d(conditioning_channels, unet.channels[0], kernel_size=3, padding=1)
        
        # Create trainable copies of the UNet's downsampling blocks
        self.control_down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(unet.channels[0], unet.channels[1]))
        self.control_down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(unet.channels[1], unet.channels[2]))
        self.control_down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(unet.channels[2], unet.channels[3]))
        self.control_bot1 = DoubleConv(unet.channels[3], unet.channels[4])
        
        # Time FiLM projections for ControlNet
        self.time_proj_inc = nn.Linear(time_dim, unet.channels[0] * 2)
        self.time_proj_down1 = nn.Linear(time_dim, unet.channels[1] * 2)
        self.time_proj_down2 = nn.Linear(time_dim, unet.channels[2] * 2)
        self.time_proj_down3 = nn.Linear(time_dim, unet.channels[3] * 2)
        self.time_proj_bot1 = nn.Linear(time_dim, unet.channels[4] * 2)
        
        # Contour FiLM projections for ControlNet
        self.contour_proj_inc = nn.Linear(contour_dim, unet.channels[0] * 2)
        self.contour_proj_down1 = nn.Linear(contour_dim, unet.channels[1] * 2)
        self.contour_proj_down2 = nn.Linear(contour_dim, unet.channels[2] * 2)
        self.contour_proj_down3 = nn.Linear(contour_dim, unet.channels[3] * 2)
        self.contour_proj_bot1 = nn.Linear(contour_dim, unet.channels[4] * 2)
        
        # Zero convolutions - these start as zero and learn to inject the control signal
        self.zero_convs = nn.ModuleList([
            nn.Conv2d(unet.channels[0], unet.channels[0], kernel_size=1),  # inc
            nn.Conv2d(unet.channels[1], unet.channels[1], kernel_size=1),  # down1
            nn.Conv2d(unet.channels[2], unet.channels[2], kernel_size=1),  # down2
            nn.Conv2d(unet.channels[3], unet.channels[3], kernel_size=1),  # down3
            nn.Conv2d(unet.channels[4], unet.channels[4], kernel_size=1),  # bot1
        ])
        
        # Initialize zero convolutions to zero
        for conv in self.zero_convs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, x, t, conditioning_signal):
        """
        Forward pass of ControlNet.
        
        Args:
            x: Input latent tensor (B, latent_dim, H, W)
            t: Timestep tensor (B,)
            conditioning_signal: Contour conditioning signal (B, 1, H, W)
            
        Returns:
            control_residuals: List of control residuals to inject into main UNet
        """
        # Resize conditioning signal to match latent space dimensions
        latent_size = x.shape[-1]  # Get the spatial size from latent tensor
        conditioning_resized = F.interpolate(conditioning_signal, size=(latent_size, latent_size), mode='bilinear', align_corners=False)
        
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Contour embedding
        c_emb = self.contour_encoder(conditioning_resized)
        
        # ControlNet branch - process conditioning signal through downsampling path
        control_hidden = self.control_conv_in(conditioning_resized)
        
        # Apply FiLM conditioning and collect residuals
        control_residuals = []
        
        # Initial conv + FiLM
        scale_t, shift_t = self.time_proj_inc(t_emb)[:, :, None, None].chunk(2, dim=1)
        scale_c, shift_c = self.contour_proj_inc(c_emb)[:, :, None, None].chunk(2, dim=1)
        control_hidden = control_hidden * (scale_t + scale_c + 1) + (shift_t + shift_c)
        control_residuals.append(self.zero_convs[0](control_hidden))
        
        # Down 1 + FiLM
        control_hidden = self.control_down1(control_hidden)
        scale_t, shift_t = self.time_proj_down1(t_emb)[:, :, None, None].chunk(2, dim=1)
        scale_c, shift_c = self.contour_proj_down1(c_emb)[:, :, None, None].chunk(2, dim=1)
        control_hidden = control_hidden * (scale_t + scale_c + 1) + (shift_t + shift_c)
        control_residuals.append(self.zero_convs[1](control_hidden))
        
        # Down 2 + FiLM
        control_hidden = self.control_down2(control_hidden)
        scale_t, shift_t = self.time_proj_down2(t_emb)[:, :, None, None].chunk(2, dim=1)
        scale_c, shift_c = self.contour_proj_down2(c_emb)[:, :, None, None].chunk(2, dim=1)
        control_hidden = control_hidden * (scale_t + scale_c + 1) + (shift_t + shift_c)
        control_residuals.append(self.zero_convs[2](control_hidden))
        
        # Down 3 + FiLM
        control_hidden = self.control_down3(control_hidden)
        scale_t, shift_t = self.time_proj_down3(t_emb)[:, :, None, None].chunk(2, dim=1)
        scale_c, shift_c = self.contour_proj_down3(c_emb)[:, :, None, None].chunk(2, dim=1)
        control_hidden = control_hidden * (scale_t + scale_c + 1) + (shift_t + shift_c)
        control_residuals.append(self.zero_convs[3](control_hidden))
        
        # Bottleneck + FiLM
        control_hidden = self.control_bot1(control_hidden)
        scale_t, shift_t = self.time_proj_bot1(t_emb)[:, :, None, None].chunk(2, dim=1)
        scale_c, shift_c = self.contour_proj_bot1(c_emb)[:, :, None, None].chunk(2, dim=1)
        control_hidden = control_hidden * (scale_t + scale_c + 1) + (shift_t + shift_c)
        control_residuals.append(self.zero_convs[4](control_hidden))
        
        return control_residuals


class ControlNetUNet(nn.Module):
    """
    Combined UNet with ControlNet for training.
    This wraps the main UNet and ControlNet together.
    """
    def __init__(self, unet: UNet2DLatent, controlnet: ControlNet):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        
    def forward(self, x, t, contour):
        """
        Forward pass with ControlNet conditioning.
        
        Args:
            x: Input latent tensor (B, latent_dim, H, W)
            t: Timestep tensor (B,)
            contour: Contour conditioning signal (B, 1, H, W)
            
        Returns:
            Predicted noise in latent space
        """
        # Get control residuals from ControlNet (use full-res contour)
        control_residuals = self.controlnet(x, t, contour)
        # Forward through main UNet with the full list of residuals
        return self.unet(x, t, control_residuals) 