import torch.nn as nn
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation


class Mask2FormerWrapper(nn.Module):
    """Hugging Face Mask2Former, returned as a plain (B,C,H,W) tensor.

    *t* is ignored just like in SwinUNetWrapper.
    """

    def __init__(self, out_channels=3, **kwargs):
        super().__init__()
        cfg = Mask2FormerConfig(num_labels=out_channels, **kwargs)
        self.net = Mask2FormerForUniversalSegmentation(cfg)

    def forward(self, x, t=None):  # noqa: D401
        # HF expects ``pixel_values`` and returns logits dict
        logits = self.net(pixel_values=x).logits  # (B, num_labels, H, W)
        return logits