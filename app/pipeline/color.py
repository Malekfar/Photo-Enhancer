"""Color / filtering primitives shared by pipeline stages. All arrays are float32."""
from __future__ import annotations

import cv2
import numpy as np

# Rec.709 luma weights applied to sRGB-encoded values (perceptual luma Y').
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
CLIP_LEVEL = 0.99


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1] + rgb[..., 2] * LUMA[2]).astype(np.float32)


def value_channel(rgb: np.ndarray) -> np.ndarray:
    return rgb.max(axis=-1)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def subsample(x: np.ndarray, max_pixels: int = 1_000_000) -> np.ndarray:
    h, w = x.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(h * w / max_pixels))))
    return x[::step, ::step]


def fast_median(y: np.ndarray) -> float:
    return float(np.median(subsample(y)))


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float, subsample_factor: int = 0) -> np.ndarray:
    """Gray-guide guided filter (He et al.). Edge-aware smoothing without halos.

    guide, src: HxW float32. subsample_factor>1 computes coefficients at reduced resolution
    (fast guided filter); 0 picks a factor automatically for large images.
    """
    guide = guide.astype(np.float32, copy=False)
    src = src.astype(np.float32, copy=False)
    h, w = guide.shape
    s = subsample_factor or max(1, min(radius // 2, int(np.sqrt(h * w / 2_000_000)) + 1))
    if s > 1:
        size = (max(1, w // s), max(1, h // s))
        g = cv2.resize(guide, size, interpolation=cv2.INTER_AREA)
        p = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
        r = max(1, radius // s)
    else:
        g, p, r = guide, src, radius
    mean_g = _box(g, r)
    mean_p = _box(p, r)
    cov_gp = _box(g * p, r) - mean_g * mean_p
    var_g = _box(g * g, r) - mean_g * mean_g
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    mean_a = _box(a, r)
    mean_b = _box(b, r)
    if s > 1:
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
    return (mean_a * guide + mean_b).astype(np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Inverse sRGB transfer; values above 1 are extended with the same curve."""
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def downscale_max_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s == 1.0:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def resize_area(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def scale_to_max_pixels(h: int, w: int, max_pixels: int) -> tuple[int, int]:
    if h * w <= max_pixels:
        return h, w
    s = np.sqrt(max_pixels / (h * w))
    return max(1, int(round(h * s))), max(1, int(round(w * s)))
