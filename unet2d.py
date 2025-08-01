
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim, bias=True),
            nn.SiLU(), # SiLU (or GELU) is standard
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

    def forward(self, t):
        return self.mlp(t)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Make sure dimensions match for skip connection
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Time embedding projection
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)


    def forward(self, x, t_emb):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        t_emb_proj = self.time_emb_proj(self.act2(t_emb))
        h = h + t_emb_proj[:, :, None, None]

        h = self.act2(self.norm2(h))
        h = self.conv2(h)

        return h + self.res_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
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
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, size, use_attention=True):
        super().__init__()
        self.channels = channels
        self.size = size
        self.use_attention = use_attention
        if self.use_attention:
            self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
            self.ln = nn.LayerNorm([channels])
            self.ff_self = nn.Sequential(
                nn.LayerNorm([channels]),
                nn.Linear(channels, channels, bias=True),
                nn.GELU(),
                nn.Linear(channels, channels, bias=True),
            )

    def forward(self, x):
        if not getattr(self, 'use_attention', True):
            return x
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

# In unet2d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Keep your existing helper classes: ---
# ResnetBlock, TimeEmbedding, SinusoidalPositionEmbeddings, SelfAttention, etc.
# ... (your other classes here) ...


class UNet2DLatent(nn.Module):
    """
    A standard, symmetric, and dimensionally correct U-Net for latent diffusion.
    This version is designed for stability and correctness.
    """
    def __init__(self, img_size, in_channels, out_channels, time_dim=256, use_attention=True):
        super().__init__()
        
        channels = [128, 256, 512, 1024]

        # --- Time Embedding ---
        self.time_embedding_dim = time_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, self.time_embedding_dim),
            nn.GELU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

        # --- Input Layer ---
        self.inc = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # --- Downsampling Path ---
        self.down1 = ResnetBlock(channels[0], channels[0], self.time_embedding_dim)
        self.down2 = ResnetBlock(channels[0], channels[1], self.time_embedding_dim)
        self.down3 = ResnetBlock(channels[1], channels[2], self.time_embedding_dim)
        self.down4 = ResnetBlock(channels[2], channels[3], self.time_embedding_dim)
        
        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bot1 = ResnetBlock(channels[3], channels[3], self.time_embedding_dim)
        self.bot_attn = SelfAttention(channels[3], img_size // 8, use_attention=use_attention)
        self.bot2 = ResnetBlock(channels[3], channels[3], self.time_embedding_dim)

        # --- Upsampling Path ---
        self.upconv4 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.up4 = ResnetBlock(channels[3], channels[2], self.time_embedding_dim) # 512 + 512 = 1024 in

        self.upconv3 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.up3 = ResnetBlock(channels[2], channels[1], self.time_embedding_dim) # 256 + 256 = 512 in
        
        self.upconv2 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.up2 = ResnetBlock(channels[1], channels[0], self.time_embedding_dim) # 128 + 128 = 256 in
        
        # --- Output Layer ---
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        
    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # --- Encoder ---
        # x shape: (B, C, 32, 32)
        x1 = self.inc(x)
        x1 = self.down1(x1, t_emb) # -> (B, 128, 32, 32)
        
        p1 = self.pool(x1) # -> (B, 128, 16, 16)
        x2 = self.down2(p1, t_emb) # -> (B, 256, 16, 16)
        
        p2 = self.pool(x2) # -> (B, 256, 8, 8)
        x3 = self.down3(p2, t_emb) # -> (B, 512, 8, 8)

        p3 = self.pool(x3) # -> (B, 512, 4, 4)
        x4 = self.down4(p3, t_emb) # -> (B, 1024, 4, 4)
        
        # --- Bottleneck ---
        x_bot = self.bot1(x4, t_emb)
        x_bot = self.bot_attn(x_bot)
        x_bot = self.bot2(x_bot, t_emb) # -> (B, 1024, 4, 4)

        # --- Decoder ---
        up4 = self.upconv4(x_bot) # -> (B, 512, 8, 8)
        up4 = torch.cat([up4, x3], dim=1) # Concat skip -> (B, 512+512=1024, 8, 8)
        up4 = self.up4(up4, t_emb) # -> (B, 512, 8, 8)
        
        up3 = self.upconv3(up4) # -> (B, 256, 16, 16)
        up3 = torch.cat([up3, x2], dim=1) # Concat skip -> (B, 256+256=512, 16, 16)
        up3 = self.up3(up3, t_emb) # -> (B, 256, 16, 16)
        
        up2 = self.upconv2(up3) # -> (B, 128, 32, 32)
        up2 = torch.cat([up2, x1], dim=1) # Concat skip -> (B, 128+128=256, 32, 32)
        up2 = self.up2(up2, t_emb) # -> (B, 128, 32, 32)
        
        return self.outc(up2)

def get_model(img_size=256, in_channels=1, out_channels=1, time_dim=None, pretrained_ckpt=None, use_attention=True):
    """
    Factory function to create the appropriate U-Net model based on input size.
    For latent space (8x8), use the simplified UNet2DLatent.
    For image space (256x256), use the full UNet2D.
    """
    if img_size <= 16:  # Latent space
        return UNet2DLatent(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            use_attention=use_attention
        )
    else:  # Image space
        return UNet2D(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained_ckpt=pretrained_ckpt,
            use_attention=use_attention
        )
