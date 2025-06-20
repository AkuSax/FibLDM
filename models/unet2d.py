import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
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
            nn.Dropout(0.1),
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
    def __init__(self, img_size=256, in_channels=1, out_channels=1, pretrained_ckpt=None):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(img_size),
            nn.Linear(img_size, img_size * 4),
            nn.GELU(),
            nn.Linear(img_size * 4, img_size),
        )
        
        # Time embedding projection layers for each stage
        time_dim = img_size
        self.time_proj_down1 = nn.Linear(time_dim, 128)
        self.time_proj_down2 = nn.Linear(time_dim, 256)
        self.time_proj_down3 = nn.Linear(time_dim, 512)
        self.time_proj_down4 = nn.Linear(time_dim, 1024)
        
        self.time_proj_up1 = nn.Linear(time_dim, 512)
        self.time_proj_up2 = nn.Linear(time_dim, 256)
        self.time_proj_up3 = nn.Linear(time_dim, 128)
        self.time_proj_up4 = nn.Linear(time_dim, 64)
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, 64)
        
        # Downsampling
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Attention layers
        self.attn1 = SelfAttention(256, size=img_size//4)
        self.attn2 = SelfAttention(512, size=img_size//8)
        self.attn3 = SelfAttention(1024, size=img_size//16)
        
        # Upsampling
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # Output convolution
        self.outc = OutConv(64, out_channels)
        
        # Load pretrained weights if provided
        if pretrained_ckpt:
            self.load_pretrained(pretrained_ckpt)
            
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Initial convolution
        x1 = self.inc(x)
        
        # Downsampling with time injection
        x2 = self.down1(x1)
        x2 = x2 + self.time_proj_down1(F.relu(t_emb))[:, :, None, None]
        
        x3 = self.down2(x2)
        x3 = x3 + self.time_proj_down2(F.relu(t_emb))[:, :, None, None]
        
        x4 = self.down3(x3)
        x4 = x4 + self.time_proj_down3(F.relu(t_emb))[:, :, None, None]
        
        x5 = self.down4(x4)
        x5 = x5 + self.time_proj_down4(F.relu(t_emb))[:, :, None, None]
        
        # Attention
        x3 = self.attn1(x3)
        x4 = self.attn2(x4)
        x5 = self.attn3(x5)
        
        # Upsampling with time injection
        x = self.up1(x5, x4)
        x = x + self.time_proj_up1(F.relu(t_emb))[:, :, None, None]
        
        x = self.up2(x, x3)
        x = x + self.time_proj_up2(F.relu(t_emb))[:, :, None, None]
        
        x = self.up3(x, x2)
        x = x + self.time_proj_up3(F.relu(t_emb))[:, :, None, None]
        
        x = self.up4(x, x1)
        x = x + self.time_proj_up4(F.relu(t_emb))[:, :, None, None]
        
        # Output convolution
        return self.outc(x)


class TD_Paint:
    def __init__(self, noise_step=1000, img_size = 256, device='cuda'):
        self.noise_step = noise_step
        self.device = device
        self.img_size = img_size
        self.t = torch.linspace(0, noise_step, noise_step+1).to(device)

        self.beta = self.cosine_beta_schedule(noise_step).to(device) # beta from t=0 to t=T
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Generates a cosine beta schedule for the given number of timesteps.

        Parameters:
        - timesteps (int): The number of timesteps for the schedule.
        - s (float): A small constant used in the calculation. Default: 0.008.

        Returns:
        - betas (torch.Tensor): The computed beta values for each timestep.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.9999)

    def sample_timesteps(self, n): # sample n timesteps from the noise schedule
        return torch.randint(low=1, high = self.noise_step, size=(n,))

    def q_sample(self, x0, tau): # forward diffusion process
        """ Forward diffusion process """
        print('alpha:', self.alpha.shape, 'tau:', tau.shape)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat.gather(0, tau.long().view(-1)).view_as(tau))
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat.gather(0, tau.long().view(-1)).view_as(tau))
        noise = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise

    def estimate_x0(self, xT, tau, predicted_noise): # estimate x0 from xT
        """ Estimate x0 from xT """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat.gather(0, tau.long().view(-1)).view_as(tau))
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat.gather(0, tau.long().view(-1)).view_as(tau))
        return (xT - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat

    def sample(self, model, n, x0, mask, edge): # reverse diffusion process, sampling 
        model.eval()
        with torch.no_grad():
            xT = torch.randn(n, 1, self.img_size, self.img_size).to(self.device) # noise image
            edge_masked = edge * (1-mask)
            x0_masked = x0 * (1 - mask)
            x_tau = xT
            for i in tqdm(reversed(range(self.noise_step)),position=0): # from T to 0
                t = (torch.ones(n) * i).long().to(self.device)
                t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                tau = t * mask
                
                x_tau = x_tau * mask + x0_masked
                predicted_noise = model(x_tau, tau, edge_masked)

                alpha = self.alpha.gather(0, tau.long().view(-1)).view_as(tau)
                beta = self.beta.gather(0, tau.long().view(-1)).view_as(tau)
                alpha_hat = self.alpha_hat.gather(0, tau.long().view(-1)).view_as(tau)

                if i>1:
                    noise = torch.randn_like(x0)
                else:
                    noise = torch.zeros_like(x0)
                x_tau = 1/torch.sqrt(alpha) * (x_tau - ((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise) + torch.sqrt(beta)*noise
        model.train()
        x_tau = (x_tau.clamp(-1,1)+1)/2
        return x_tau


class TD_Paint_v2: # adding clean image as initial condition
    def __init__(self, noise_step=1000, img_size = 256, device='cuda'):
        self.noise_step = noise_step
        self.device = device
        self.img_size = img_size
        self.t = torch.linspace(0, noise_step, noise_step+1).to(device)

        self.beta = self.cosine_beta_schedule(noise_step).to(device) # beta from t=0 to t=T
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Generates a cosine beta schedule for the given number of timesteps.

        Parameters:
        - timesteps (int): The number of timesteps for the schedule.
        - s (float): A small constant used in the calculation. Default: 0.008.

        Returns:
        - betas (torch.Tensor): The computed beta values for each timestep.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.9999)

    def sample_timesteps(self, n): # sample n timesteps from the noise schedule
        return torch.randint(low=1, high = self.noise_step, size=(n,))

    def q_sample(self, x0, tau): # forward diffusion process
        """ Forward diffusion process """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat.gather(0, tau.long().view(-1)).view_as(tau))
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat.gather(0, tau.long().view(-1)).view_as(tau))
        noise = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise

    def sample(self, model, n, x0, mask, edge): # reverse diffusion process, sampling 
        model.eval()
        with torch.no_grad():
            xT = torch.randn(n, 1, self.img_size, self.img_size).to(self.device) # noise image
            edge_masked = edge * (1-mask)
            x0_masked = x0 * (1 - mask)
            x_tau = xT
            for i in tqdm(reversed(range(self.noise_step)),position=0): # from T to 0
                t = (torch.ones(n) * i).long().to(self.device)
                t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                tau = t * mask
                
                x_tau = x_tau * mask + x0_masked
                predicted_noise = model(x_tau, tau, edge_masked)

                alpha = self.alpha.gather(0, tau.long().view(-1)).view_as(tau)
                beta = self.beta.gather(0, tau.long().view(-1)).view_as(tau)
                alpha_hat = self.alpha_hat.gather(0, tau.long().view(-1)).view_as(tau)

                if i>1:
                    noise = torch.randn_like(x0)
                else:
                    noise = torch.zeros_like(x0)
                x_tau = 1/torch.sqrt(alpha) * (x_tau - ((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise) + torch.sqrt(beta)*noise
        model.train()
        x_tau = (x_tau.clamp(-1,1)+1)/2
        return x_tau
