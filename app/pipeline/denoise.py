"""Denoising stage: noise estimation, SCUNet (official real-image weights) and OpenCV NLM."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

from app.pipeline.color import guided_filter, luminance, smoothstep, subsample
from app.pipeline.tiling import tiled_apply, to_numpy, to_tensor

_LAPLACE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def suppress_amplified_chroma(
    base: np.ndarray,
    out: np.ndarray,
    *,
    gain_low: float,
    gain_high: float,
    radius: int,
    eps: float,
) -> tuple[np.ndarray, float]:
    """Smooth chromaticity where the enhancement applied a strong local gain.

    A global noise estimate misses dark regions that are brightened far more than the
    image as a whole; there, color noise and JPEG chroma blocks get amplified into blotches.
    Chromaticity (rgb / Y) is smoothed edge-aware (guided by the output luminance) and
    blended in with weight smoothstep(gain_low, gain_high, local gain). Luminance, and so
    all structure, is unchanged. Returns the image and the share of pixels affected.
    """
    y_b = luminance(base)
    y_o = luminance(out)
    num = guided_filter(y_b, y_o, 4, 1e-4)
    den = guided_filter(y_b, y_b, 4, 1e-4)
    gain = np.maximum(num, 0.0) / (np.maximum(den, 0.0) + 2e-3)
    w = smoothstep(gain_low, gain_high, gain)
    if float(w.max()) < 1e-3:
        return out, 0.0
    e = 1e-3
    chroma = out / (y_o[..., None] + e)
    smooth = np.stack([guided_filter(y_o, chroma[..., c], radius, eps) for c in range(3)], axis=-1)
    smooth = np.clip(smooth, 0.0, None)
    mixed = w[..., None] * smooth + (1.0 - w[..., None]) * chroma
    res = mixed * (y_o[..., None] + e)
    # keep luminance exact: rescale so Y(res) == Y(out)
    res *= (y_o / np.maximum(luminance(res), 1e-6))[..., None]
    return np.clip(res, 0.0, None).astype(np.float32), float(np.mean(w > 0.5))


def estimate_noise_sigma(rgb: np.ndarray) -> float:
    """Immerkær (1996) fast noise std estimate on luminance, in [0,1] units.

    Uses a central crop for very large images (noise is spatially stationary enough and the
    estimate must stay cheap).
    """
    y = luminance(rgb)
    h, w = y.shape
    max_side = 2048
    if h > max_side or w > max_side:
        y0, x0 = max(0, (h - max_side) // 2), max(0, (w - max_side) // 2)
        y = y[y0 : y0 + max_side, x0 : x0 + max_side]
        h, w = y.shape
    if h < 3 or w < 3:
        return 0.0
    conv = cv2.filter2D(y, -1, _LAPLACE_KERNEL, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


def _normalisation_scale(rgb: np.ndarray) -> float:
    """Scale that stretches a dark image to use the full range before denoising.

    Denoisers are tuned for normally exposed content; stretching makes the noise level of
    a dark image look like what the model was trained on, and avoids 8-bit quantisation
    of shadows for OpenCV.
    """
    p = float(np.percentile(subsample(rgb.max(axis=-1)), 99.5))
    return 1.0 / max(p, 0.05)


class Denoiser(ABC):
    name: str

    @abstractmethod
    def _denoise_normalised(self, x: np.ndarray, sigma: float) -> np.ndarray: ...

    def denoise(self, rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
        if strength <= 0.0:
            return rgb
        s = _normalisation_scale(rgb)
        x = np.clip(rgb * s, 0.0, 1.0)
        sigma = estimate_noise_sigma(x)
        den = self._denoise_normalised(x, sigma)
        # Only the residual is transferred back: un-stretched pixels (>1 after scaling)
        # and sub-quantum precision of the input are preserved.
        out = rgb + strength * (den - x) / s
        return np.clip(out, 0.0, 1.0).astype(np.float32)


class OpenCVDenoiser(Denoiser):
    name = "opencv"

    def _denoise_normalised(self, x: np.ndarray, sigma: float) -> np.ndarray:
        u8 = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
        h = float(np.clip(sigma * 255.0 * 1.2, 3.0, 25.0))
        den = cv2.fastNlMeansDenoisingColored(
            cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), None, h, h, 7, 21
        )
        den = cv2.cvtColor(den, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Residual relative to the *quantised* input, so quantisation error is not added.
        return x + (den - u8.astype(np.float32) / 255.0)


class SCUNetDenoiser(Denoiser):
    name = "scunet"

    def __init__(self, model: torch.nn.Module, device: torch.device, tile: int, overlap: int) -> None:
        self.model = model
        self.device = device
        self.tile = tile
        self.overlap = overlap
        self._multiple = getattr(model, "size_multiple", 64)
        self._lock = threading.Lock() if device.type == "cuda" else None

    def _forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.model(t.to(self.device))

    def _denoise_normalised(self, x: np.ndarray, sigma: float) -> np.ndarray:
        with torch.inference_mode():
            t = to_tensor(x, torch.device("cpu"))
            if self._lock is not None:
                with self._lock:
                    out = tiled_apply(self._forward, t, self.tile, self.overlap, self._multiple)
            else:
                out = tiled_apply(self._forward, t, self.tile, self.overlap, self._multiple)
        return np.clip(to_numpy(out), 0.0, 1.0)


def should_denoise(
    mode: str,
    sigma: float,
    median_luma: float,
    *,
    dark_threshold: float,
    noise_threshold: float,
    target_median: float,
) -> tuple[bool, float]:
    """Decide whether to denoise. Returns (apply, projected_noise).

    projected_noise is the noise std expected *after* brightening the median to the
    target: sigma * target / median. Brightening amplifies noise by about that factor.
    """
    projected = sigma * target_median / max(median_luma, 0.02)
    if mode == "on":
        return True, projected
    if mode == "off":
        return False, projected
    return (median_luma < dark_threshold and projected > noise_threshold), projected
