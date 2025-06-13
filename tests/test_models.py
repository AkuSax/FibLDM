import torch
import pytest

from models import MODEL_REGISTRY, get_model

DUMMY = torch.randn(2, 3, 32, 32)
TIMES = torch.randint(0, 1000, (2,))


@pytest.mark.parametrize("arch", list(MODEL_REGISTRY))
def test_forward_pass(arch):
    model = get_model(arch, img_size=32, in_channels=3, out_channels=3)
    out = model(DUMMY, TIMES)
    assert out.shape == DUMMY.shape, f"{arch} produced {out.shape}, expected {DUMMY.shape}"