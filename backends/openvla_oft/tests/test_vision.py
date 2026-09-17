import numpy as np
import pytest
from efficient_vla_backend_openvla_oft.vision import patch_cosine_similarity


def test_patch_cosines_follow_row_major_tokens_without_mutating_images():
    current = np.zeros((224, 224, 3), dtype=np.uint8)
    current[:, :, 0] = 100
    previous = current.copy()
    previous[14:28, 28:42] = [0, 0, 100]
    expected = np.ones(256)
    expected[18] = 0
    np.testing.assert_allclose(patch_cosine_similarity(current, previous), expected, atol=2e-7)
    np.testing.assert_array_equal(current[14, 28], [100, 0, 0])
    np.testing.assert_array_equal(previous[14, 28], [0, 0, 100])


def test_patch_geometry_rejects_uncropped_images():
    with pytest.raises(ValueError, match="preprocessed"):
        patch_cosine_similarity(np.zeros((256, 256, 3)), np.zeros((256, 256, 3)))
