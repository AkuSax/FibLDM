import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, size=None):
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        if self.size is None:
            raise ValueError("SelfAttention: 'size' must be specified (got None)")
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value)
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        # --- Channel Attention ---
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Avg Pool
            nn.Conv2d(in_channels, in_channels // reduction, 1),  # Squeeze
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),  # Excitation
            nn.Sigmoid()
        )

        # --- Spatial Attention ---
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply Channel Attention
        ch_attn = self.channel_attention(x)  # Shape: [B, C, 1, 1]
        x = x * ch_attn  # Apply channel-wise scaling

        # Compute Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Avg pool across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pool across channels
        spatial_attn = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))  # [B, 2, H, W]

        # Apply Spatial Attention
        return x * spatial_attn  # Element-wise multiplication
    

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class UNet2D(nn.Module):
    def __init__(self, img_size=256, in_channels=1, out_channels=1, 
                 channels=[64, 128, 256, 512, 1024], pretrained_ckpt=None):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(img_size),
            nn.Linear(img_size, img_size * 4),
            nn.GELU(),
            nn.Linear(img_size * 4, img_size),
        )
        
        # Time embedding projection layers for each stage
        time_dim = img_size
        self.time_proj_down1 = nn.Linear(time_dim, channels[1] * 2)
        self.time_proj_down2 = nn.Linear(time_dim, channels[2] * 2)
        self.time_proj_down3 = nn.Linear(time_dim, channels[3] * 2)
        self.time_proj_down4 = nn.Linear(time_dim, channels[4] * 2)
        
        self.time_proj_up1 = nn.Linear(time_dim, channels[3] * 2)
        self.time_proj_up2 = nn.Linear(time_dim, channels[2] * 2)
        self.time_proj_up3 = nn.Linear(time_dim, channels[1] * 2)
        self.time_proj_up4 = nn.Linear(time_dim, channels[0] * 2)
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, channels[0])
        
        # Downsampling
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])
        
        # Attention layers
        self.attn1 = SelfAttention(channels[2], size=img_size//4)
        self.attn2 = SelfAttention(channels[3], size=img_size//8)
        self.attn3 = SelfAttention(channels[4], size=img_size//16)
        
        # Upsampling
        self.up1 = Up(channels[4], channels[3])
        self.up2 = Up(channels[3], channels[2])
        self.up3 = Up(channels[2], channels[1])
        self.up4 = Up(channels[1], channels[0])
        
        # Output convolution
        self.outc = OutConv(channels[0], out_channels)
        
        # Load pretrained weights if provided
        if pretrained_ckpt:
            self.load_pretrained(pretrained_ckpt)
            
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Initial convolution
        x1 = self.inc(x)
        
        # Downsampling with FiLM time injection
        x2 = self.down1(x1)
        scale, shift = self.time_proj_down1(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x2 = x2 * (scale + 1) + shift
        
        x3 = self.down2(x2)
        scale, shift = self.time_proj_down2(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x3 = x3 * (scale + 1) + shift
        
        x4 = self.down3(x3)
        scale, shift = self.time_proj_down3(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x4 = x4 * (scale + 1) + shift
        
        x5 = self.down4(x4)
        scale, shift = self.time_proj_down4(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x5 = x5 * (scale + 1) + shift
        
        # Attention
        x3 = self.attn1(x3)
        x4 = self.attn2(x4)
        x5 = self.attn3(x5)
        
        # Upsampling with FiLM time injection
        x = self.up1(x5, x4)
        scale, shift = self.time_proj_up1(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x = x * (scale + 1) + shift
        
        x = self.up2(x, x3)
        scale, shift = self.time_proj_up2(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x = x * (scale + 1) + shift
        
        x = self.up3(x, x2)
        scale, shift = self.time_proj_up3(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x = x * (scale + 1) + shift
        
        x = self.up4(x, x1)
        scale, shift = self.time_proj_up4(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x = x * (scale + 1) + shift
        
        # Output convolution
        return self.outc(x)


class UNet2DLatent(nn.Module):
    """
    Final, optimized U-Net for latent space.
    - Uses the standard 'Up' module with learnable ConvTranspose2d for upsampling.
    - This version has the most expressive power for the given stable training environment.
    """
    def __init__(self, img_size, in_channels, out_channels):
        super().__init__()
        print(f"[UNet2DLatent] Initializing with FINAL architecture.")
        time_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.channels = [64, 128, 256, 512, 1024]
        # Time FiLM projections for all blocks
        self.time_proj_inc = nn.Linear(time_dim, self.channels[0] * 2)
        self.time_proj_down1 = nn.Linear(time_dim, self.channels[1] * 2)
        self.time_proj_down2 = nn.Linear(time_dim, self.channels[2] * 2)
        self.time_proj_down3 = nn.Linear(time_dim, self.channels[3] * 2)
        self.time_proj_bot1 = nn.Linear(time_dim, self.channels[4] * 2)
        self.time_proj_up1 = nn.Linear(time_dim, self.channels[3] * 2)
        self.time_proj_up2 = nn.Linear(time_dim, self.channels[2] * 2)
        self.time_proj_up3 = nn.Linear(time_dim, self.channels[1] * 2)
        # Downsampling path
        self.inc = DoubleConv(in_channels, self.channels[0])
        self.down1 = Down(self.channels[0], self.channels[1])
        self.down2 = Down(self.channels[1], self.channels[2])
        self.down3 = Down(self.channels[2], self.channels[3])
        # Bottleneck
        self.bot1 = DoubleConv(self.channels[3], self.channels[4])
        # Upsampling path using the 'Up' module with ConvTranspose2d
        self.up1 = Up(self.channels[4], self.channels[3])
        self.up2 = Up(self.channels[3], self.channels[2])
        self.up3 = Up(self.channels[2], self.channels[1])
        self.up4 = Up(self.channels[1], self.channels[0])
        self.outc = nn.Conv2d(self.channels[0], out_channels, kernel_size=1)
    def forward(self, x, t, control_residuals=None):
        if control_residuals is not None:
            raise NotImplementedError("ControlNet path needs to be adapted for this UNet version.")
        t_emb = self.time_mlp(t)
        # --- Downsampling with FiLM ---
        x1 = self.inc(x)
        scale, shift = self.time_proj_inc(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x1 = x1 * (scale + 1) + shift
        x2 = self.down1(x1)
        scale, shift = self.time_proj_down1(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x2 = x2 * (scale + 1) + shift
        x3 = self.down2(x2)
        scale, shift = self.time_proj_down2(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x3 = x3 * (scale + 1) + shift
        x4 = self.down3(x3)
        scale, shift = self.time_proj_down3(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x4 = x4 * (scale + 1) + shift
        # --- Bottleneck with FiLM ---
        x5 = self.bot1(x4)
        scale, shift = self.time_proj_bot1(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x5 = x5 * (scale + 1) + shift
        # --- Upsampling with FiLM ---
        x = self.up1(x5, x4)
        scale, shift = self.time_proj_up1(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x = x * (scale + 1) + shift
        x = self.up2(x, x3)
        scale, shift = self.time_proj_up2(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x = x * (scale + 1) + shift
        x = self.up3(x, x2)
        scale, shift = self.time_proj_up3(F.relu(t_emb))[:, :, None, None].chunk(2, dim=1)
        x = x * (scale + 1) + shift
        x = self.up4(x, x1)
        out = self.outc(x)
        return out


def get_model(img_size=256, in_channels=1, out_channels=1, time_dim=None, pretrained_ckpt=None):
    """
    Factory function to create the appropriate U-Net model based on input size.
    For latent space (8x8), use the simplified UNet2DLatent.
    For image space (256x256), use the full UNet2D.
    """
    if img_size <= 16:  # Latent space
        return UNet2DLatent(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels
        )
    else:  # Image space
        return UNet2D(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained_ckpt=pretrained_ckpt
        )
