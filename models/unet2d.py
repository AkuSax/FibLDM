import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        if self.residual:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # print(x.shape)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # print('emb:', emb.shape)
        return x+emb
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim = 256):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            DoubleConv(in_channels*2, in_channels*2, residual=True),
            DoubleConv(in_channels*2, out_channels, mid_channels=in_channels)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x+emb

class SelfAttention(nn.Module):
    def __init__(self, in_channels, size):
        super().__init__()
        self.in_channels = in_channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(in_channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.size*self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2,1).view(-1, self.in_channels, self.size, self.size)

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
    

class UNet2D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_dim = 256, device = None): 
        super(UNet2D, self).__init__()
        self.time_dim = time_dim
        self.device = device
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.att1 = CBAM(128, reduction=16, kernel_size=7)

        self.down2 = Down(128, 256)
        self.att2 = CBAM(256, reduction=8, kernel_size=5)

        self.down3 = Down(256, 512)
        self.att3 = CBAM(512, reduction=4, kernel_size=3)

        # Bottleneck
        self.bot1 = DoubleConv(512, 1024)
        self.bot2 = DoubleConv(1024, 1024)
        self.bot3 = DoubleConv(1024, 256)
    
        # Decoder
        self.up1 = Up(256, 128)
        self.att4 = CBAM(128, reduction=8, kernel_size=5)

        self.up2 = Up(128, 64)
        self.att5 = CBAM(64, reduction=8, kernel_size=5)

        self.up3 = Up(64, 64)
        self.att6 = CBAM(64, reduction=4, kernel_size=7)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def pos_encoding(self, t, channels): # Positional Encoding of the time dimension
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels)).to(self.device)
        pos_enc_a = torch.sin(t * inv_freq)
        pos_enc_b = torch.cos(t * inv_freq)
        pos_enc = torch.cat((pos_enc_a, pos_enc_b), dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float32)
        t = self.pos_encoding(t, self.time_dim)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.att1(x2)
        x3 = self.down2(x2, t)
        x3 = self.att2(x3)
        x4 = self.down3(x3, t)
        x4 = self.att3(x4)

        x5 = self.bot1(x4)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up1(x5, x3, t)
        x = self.att4(x)
        x = self.up2(x, x2, t)
        x = self.att5(x)
        x = self.up3(x, x1, t)
        x = self.att6(x)
        x = self.outc(x)
        return x


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
