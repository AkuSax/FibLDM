import torch
from torch import nn
from monai.networks.nets import SwinUNETR

GITHUB_CKPT = (
    "https://github.com/Project-MONAI/MONAI/"
    "releases/download/v1.3.0/monai_swin_unetr-589a67d7.pth"
)

class SwinUNetWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int,          
        out_channels: int,  
        img_size: int | None = None,
        pretrained_ckpt: str | None = None,
    ):
        super().__init__()
        self.net = SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            norm_name="instance",
            dropout_path_rate=0.0,
        )

        

        if pretrained_ckpt:
            url = pretrained_ckpt or GITHUB_CKPT
            ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False)
            sd = ckpt.get("state_dict", ckpt)
            self.net.load_state_dict(sd, strict=False)


    def forward(self, x, t):
        x = x.unsqueeze(2)      # (B, C, H, W) → (B, C, D=1, H, W)
        out = self.net(x)       # → (B, out_channels, 1, H, W)
        return out.squeeze(2)   # → (B, out_channels, H, W)
