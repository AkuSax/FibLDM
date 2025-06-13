import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    def __init__(
        self,
        device: torch.device,
        in_channels: int = 1,
        ndf: int = 64,
        n_layers: int = 3
    ):
        super().__init__()
        # Initial conv
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Down‐sampling blocks
        nf_mult = 1
        for layer in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**layer, 8)
            sequence += [
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf*nf_mult, affine=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Penultimate layer (stride=1)
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult,
                      kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*nf_mult, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Final output layer: 1‐channel logits map
        sequence += [
            nn.Conv2d(ndf*nf_mult, 1, kernel_size=4, stride=1, padding=1, bias=True)
        ]
        
        self.model = nn.Sequential(*sequence)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
