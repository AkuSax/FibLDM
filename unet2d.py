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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
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
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).swapaxes(1, 2)  # (b, hw, c)
        x_ln = self.ln(x_flat)
        attn, _ = self.mha(x_ln, x_ln, x_ln)
        attn = attn + x_flat
        attn = self.ff_self(attn)
        attn = attn.swapaxes(2, 1).view(b, c, h, w)
        return attn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class UNet2DLatent(nn.Module):
    """
    ✨ Corrected UNet with Self-Attention for a 16x16 latent space. ✨
    This version is architecturally symmetric.
    """
    def __init__(self, img_size, in_channels, out_channels, time_dim=256):
        super().__init__()
        self.img_size = img_size
        self.time_dim = time_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4), nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Downsampling: 16x16 -> 8x8
        self.inc = DoubleConv(in_channels, 128)
        self.down1 = Down(128, 256)
        self.attn1 = SelfAttention(256, img_size // 2)

        # Bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.attn_bot = SelfAttention(512, img_size // 2)
        self.bot2 = DoubleConv(512, 256)

        # Upsampling: 8x8 -> 16x16
        self.up1 = Up(256, 128)
        self.attn2 = SelfAttention(128, img_size)
        self.outc = OutConv(128, out_channels)

        # Time injection layers (additive)
        self.time_proj_inc = nn.Linear(time_dim, 128)
        self.time_proj_down1 = nn.Linear(time_dim, 256)
        self.time_proj_bot = nn.Linear(time_dim, 256)
        self.time_proj_up1 = nn.Linear(time_dim, 128)
        
    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Downsampling
        x1 = self.inc(x) # 16x16, 128ch
        x1 = x1 + self.time_proj_inc(t_emb)[:, :, None, None]

        x2 = self.down1(x1) # 8x8, 256ch
        x2 = x2 + self.time_proj_down1(t_emb)[:, :, None, None]
        x2 = self.attn1(x2)
        
        # Bottleneck
        x_bot = self.bot1(x2) # 8x8, 512ch
        x_bot = self.attn_bot(x_bot)
        x_bot = self.bot2(x_bot) # 8x8, 256ch
        x_bot = x_bot + self.time_proj_bot(t_emb)[:, :, None, None]

        # Upsampling
        x = self.up1(x_bot, x1) 
        x = x + self.time_proj_up1(t_emb)[:, :, None, None]
        x = self.attn2(x)
        
        return self.outc(x)


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
        self.attn1 = SelfAttention(channels[2], img_size//4)
        self.attn2 = SelfAttention(channels[3], img_size//8)
        self.attn3 = SelfAttention(channels[4], img_size//16)
        
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
