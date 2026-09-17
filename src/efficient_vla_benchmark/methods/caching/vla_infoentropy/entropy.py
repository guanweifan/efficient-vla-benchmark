"""Image entropy and fixed-execution-time allocation; no model dependencies."""

import math

import numpy as np


def image_entropy(image):
    rgb = np.asarray(image)
    if rgb.shape != (224, 224, 3):
        raise ValueError("Image entropy requires policy-preprocessed RGB patches")
    # Explicit BT.601 convention, 256 bins; no trained saliency estimator.
    gray = np.rint(rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
    gray = gray.clip(0, 255).astype(np.int64)
    patches = gray.reshape(16, 14, 16, 14).transpose(0, 2, 1, 3).reshape(256, -1)
    bins = patches + 256 * np.arange(256)[:, None]
    counts = np.bincount(bins.ravel(), minlength=256 * 256).reshape(256, 256)
    probability = counts / 196
    logarithm = np.zeros_like(probability)
    np.log2(probability, out=logarithm, where=probability > 0)
    return -(probability * logarithm).sum(-1) / 8


def quotas(step, low, high, transition):
    alpha = min(1.0, max(0.0, step / transition))
    return math.floor(high - (high - low) * alpha), math.floor(low + (high - low) * alpha)
