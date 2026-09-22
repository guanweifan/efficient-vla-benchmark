from types import SimpleNamespace

import numpy as np
import pytest
import torch
from efficient_vla_backend_cogact.vision import policy_image
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)


@pytest.mark.parametrize("crop", [False, True])
def test_policy_rgb_matches_native_spatial_preprocessing(crop):
    image = Image.fromarray(
        np.random.default_rng(3).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    )
    spatial = [Resize(240 if crop else (224, 224), interpolation=InterpolationMode.BICUBIC)]
    if crop:
        spatial.append(CenterCrop(224))
    tensor = [ToTensor(), Normalize([0.5] * 3, [0.5] * 3)]
    pipeline = Compose([*spatial, *tensor])
    vision = SimpleNamespace(image_transform=SimpleNamespace(siglip_image_transform=pipeline))
    actual = Compose(tensor)(policy_image(vision, image))
    torch.testing.assert_close(actual, pipeline(image), atol=0, rtol=0)
