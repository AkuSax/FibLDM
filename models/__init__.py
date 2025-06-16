from .unet2d import UNet2D
from .swin_unet import SwinUNetWrapper
from .mask2former import Mask2FormerWrapper

MODEL_REGISTRY = {
    "unet2d":      UNet2D,
    "swin_unet":   SwinUNetWrapper,
    "mask2former": Mask2FormerWrapper,
}

def get_model(arch, img_size=256, in_channels=1, out_channels=1, pretrained_ckpt=None):
    if arch == "unet2d":
        return UNet2D(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained_ckpt=pretrained_ckpt
        )
    elif arch == "swin_unet":
        return SwinUNetWrapper(**kwargs)
    elif arch == "mask2former":
        return Mask2FormerWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
