import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=8):
        super().__init__()
        # A simple convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
            # The output of this layer is (batch_size, latent_dim * 2, 16, 16)
            nn.Conv2d(512, latent_dim * 2, kernel_size=4, stride=2, padding=1) # 16 -> 8
        )

        # A corresponding decoder
        self.decoder = nn.Sequential(
            # Input is (batch_size, latent_dim, 8, 8)
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1), # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1), # 128 -> 256
            nn.Tanh() # To output in [-1, 1] range like your original data
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar 