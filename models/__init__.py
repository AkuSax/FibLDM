from .unet2d import UNet2D
from .swin_unet import SwinUNetWrapper
from .mask2former import Mask2FormerWrapper

MODEL_REGISTRY = {
    "unet2d":      UNet2D,
    "swin_unet":   SwinUNetWrapper,
    "mask2former": Mask2FormerWrapper,
}

def get_model(name: str, **kwargs):
    name = name.lower()
    if name == "unet2d":
        return UNet2D(
            in_channels=kwargs.get("in_channels", 1),
            out_channels=kwargs.get("out_channels", 1)
        )
    elif name == "swin_unet":
        return SwinUNetWrapper(**kwargs)
    elif name == "mask2former":
        return Mask2FormerWrapper(**kwargs)
    else:
        raise KeyError(f"Unknown architecture '{name}'. Available: {list(MODEL_REGISTRY)}")
