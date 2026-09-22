"""Reversible timm probes and fixed-grid RGB geometry, without paper decisions."""

import numpy as np


class VisionAttentionProbe:
    """Capture normalized timm Q/K for each view without changing attention output.

    The consumer chooses the encoder/block and owns every scoring decision. Stored
    tensors last only until take()/clear(); close restores the unmodified instance.
    """

    def __init__(self, attention):
        self.clear()
        self.handles = [
            attention.q_norm.register_forward_hook(self._query),
            attention.k_norm.register_forward_hook(self._key),
        ]

    def _query(self, module, args, output):
        self.queries.append(output.detach())

    def _key(self, module, args, output):
        self.keys.append(output.detach())

    def clear(self):
        self.queries, self.keys = [], []

    def take(self):
        if len(self.queries) != len(self.keys):
            self.clear()
            raise RuntimeError("Incomplete vision Q/K probe")
        result = list(zip(self.queries, self.keys))
        self.clear()
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


def patch_cosine_similarity(current, previous):
    """Cosines of row-major 14x14 RGB patches in policy-preprocessed images."""

    def patches(image):
        array = np.asarray(image)
        if array.shape != (224, 224, 3):
            raise ValueError("Patch comparison requires preprocessed 224x224 RGB images")
        return (
            array.reshape(16, 14, 16, 14, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(256, -1)
            .astype(np.float32)
        )

    a, b = patches(current), patches(previous)
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
