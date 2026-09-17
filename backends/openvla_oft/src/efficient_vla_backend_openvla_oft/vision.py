"""Model-specific patch geometry shared by visual-token methods."""

import numpy as np


def patch_cosine_similarity(current, previous):
    """Cosines of row-major 14x14 RGB patches in policy-preprocessed images."""

    def patches(image):
        array = np.asarray(image)
        if array.shape != (224, 224, 3):
            raise ValueError("OFT patch comparison requires preprocessed 224x224 RGB images")
        return (
            array.reshape(16, 14, 16, 14, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(256, -1)
            .astype(np.float32)
        )

    a, b = patches(current), patches(previous)
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
