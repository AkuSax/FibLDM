from .unet2d import UNet2D
from .swin_unet import SwinUNetWrapper
from .mask2former import Mask2FormerWrapper

MODEL_REGISTRY = {
    "unet2d":      UNet2D,
    "swin_unet":   SwinUNetWrapper,
    "mask2former": Mask2FormerWrapper,
}

def get_model(name: str, **kwargs):
    try:
        ModelCls = MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown architecture '{name}'. Available: {list(MODEL_REGISTRY)}")
    return ModelCls(**kwargs)
