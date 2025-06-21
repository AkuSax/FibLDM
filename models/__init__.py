from .unet2d import UNet2D

def get_model(img_size=256, in_channels=1, out_channels=1, pretrained_ckpt=None, **kwargs):
    return UNet2D(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained_ckpt=pretrained_ckpt
    )
