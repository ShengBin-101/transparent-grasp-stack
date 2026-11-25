"""The photometric augmentations the segmenter trains with (segmenter/data.py::augment).

From the project's training repository (sgdc/data/augment.py at commit 95c3499), unchanged
apart from one docstring, so a rerun of segmenter/train.py draws the same augmentations.
"""
import cv2
import numpy as np


def chromatic_jitter(rgb: np.ndarray, rng: np.random.Generator,
                     strength: float = 0.1) -> np.ndarray:
    """Hue / lightness / saturation jitter in HLS, as the TransCG loaders do."""
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS).astype(np.float32)
    d = (rng.random(3) - 0.5) * strength * 180.0
    hls[..., 0] = (hls[..., 0] + d[0]) % 180
    hls[..., 1] = np.clip(hls[..., 1] + d[1], 0, 255)
    hls[..., 2] = np.clip(hls[..., 2] + d[2], 0, 255)
    return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2RGB)


def photometric_noise(rgb: np.ndarray, rng: np.random.Generator,
                      level: float = 0.1) -> np.ndarray:
    """Gaussian noise, or occasionally motion blur."""
    if rng.random() < 0.9:
        sigma = rng.random() * level * 256
        noise = rng.normal(0.0, max(sigma, 1e-6), size=rgb.shape[:2]).astype(np.float32)
        out = rgb.astype(np.float32) + noise[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)
    size = int(rng.choice([3, 5, 7, 9, 11, 15]))
    k = np.zeros((size, size), np.float32)
    if rng.random() < 0.5:
        k[size // 2, :] = 1.0
    else:
        k[:, size // 2] = 1.0
    return cv2.filter2D(rgb, -1, k / size)


def brightness_scale(rgb: np.ndarray, rng: np.random.Generator,
                     lo: float = 0.35, hi: float = 1.4) -> np.ndarray:
    """Global gain.

    The rig's first failed capture had a mean RGB brightness of 15/255, and TransCG
    is uniformly well lit, so training shows the head dark frames too.
    """
    g = rng.uniform(lo, hi)
    return np.clip(rgb.astype(np.float32) * g, 0, 255).astype(np.uint8)
