"""Final adjustments: auto-exposure, mild unsharp mask, strength blend."""
from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.color import linear_to_srgb, luminance, srgb_to_linear, subsample


def pre_exposure(rgb: np.ndarray, target: float | None, max_gain: float, knee: float) -> tuple[np.ndarray, dict]:
    """Raise exposure in linear light so the model sees input in its trained range.

    Equivalent to a longer exposure: a global multiplication of scene radiance, so it adds
    no structure. The gain is chosen so the sRGB median luminance reaches `target`; values
    pushed above 1 are rolled off with the soft-knee instead of clipped. Never darkens.
    """
    from app.pipeline.highlight import soft_knee  # noqa: PLC0415 - avoid import cycle

    y_s = subsample(luminance(rgb), 250_000)
    # 8-bit near-black frames can have a median of exactly 0; measure a brighter quantile then.
    q = 50.0 if float(np.median(y_s)) >= 1.0 / 255.0 else 75.0
    med = float(np.percentile(y_s, q))
    info = {"median_before": round(float(np.median(y_s)), 5), "gain": 1.0}
    if target is None or med >= target or med <= 0.0:
        return rgb, info

    lin_s = srgb_to_linear(subsample(rgb, 250_000))

    def median_at(g: float) -> float:
        return float(np.percentile(luminance(np.minimum(linear_to_srgb(lin_s * g), 1.0)), q))

    lo, hi = 0.0, float(np.log(max_gain))
    if median_at(max_gain) < target:
        gain = max_gain
    else:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if median_at(float(np.exp(mid))) < target else (lo, mid)
        gain = float(np.exp(hi))
    out, _ = soft_knee(linear_to_srgb(srgb_to_linear(rgb) * gain), knee)
    info["gain"] = round(float(gain), 4)
    return out, info


def background_level(rgb: np.ndarray, min_fraction: float) -> float | None:
    """Luminance of a seamless studio background, or None if there is none.

    Background = flat (no texture), neutral (low chroma) pixels covering at least
    `min_fraction` of the frame and brighter than the scene median.
    """
    small = cv2.resize(rgb, (256, max(1, round(256 * rgb.shape[0] / rgb.shape[1]))), interpolation=cv2.INTER_AREA)
    y = luminance(small)
    chroma = small.max(axis=-1) - small.min(axis=-1)
    texture = np.abs(y - cv2.GaussianBlur(y, (0, 0), 2.0))
    med = float(np.median(y))
    flat_neutral = (chroma < 0.06 + 0.1 * y) & (texture < 0.01 + 0.03 * y) & (y >= med) & (y > 0.02)
    if flat_neutral.mean() < min_fraction:
        return None
    return float(np.median(y[flat_neutral]))


def _masked_gain(gain: float, mask: np.ndarray) -> np.ndarray:
    return 1.0 + (gain - 1.0) * (1.0 - mask)


def auto_exposure(
    rgb: np.ndarray,
    mask: np.ndarray,
    target_low: float,
    target_high: float,
    max_gain: float,
    allow_darken: bool,
    white_target: float | None = None,
    bg_min_fraction: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """Scale so the median luminance lands in [target_low, target_high].

    With `white_target`, if a seamless background is detected (see `background_level`),
    the gain is also at least what brings it to `white_target`: in a dark white-background
    product shot the median *is* the background, and a median target alone leaves it gray.

    The gain is spatially weighted by (1 - highlight mask), so protected highlights are
    never pushed. Values above 1 are rolled off by the final highlight guard. By default
    the image is never darkened (a white-background product shot has a high median by
    design and must stay as is).
    """
    y_s = subsample(luminance(rgb), 500_000)
    m_s = subsample(mask, 500_000)
    med = float(np.median(y_s))
    info = {"median_before": med, "gain": 1.0}

    white_gain = 1.0
    if white_target is not None:
        bg = background_level(rgb, bg_min_fraction)
        info["background_level"] = bg if bg is not None else -1.0
        if bg is not None:
            white_gain = float(np.clip(white_target / max(bg, 1e-4), 1.0, max_gain))
    info["white_gain"] = white_gain

    if med > target_high and not allow_darken:
        return _apply_masked(rgb, mask, white_gain, info)
    if target_low <= med <= target_high:
        return _apply_masked(rgb, mask, white_gain, info)

    target = target_low if med < target_low else target_high

    def median_at(g: float) -> float:
        return float(np.median(np.minimum(y_s * _masked_gain(g, m_s), 1.0)))

    if med < target_low:
        lo, hi = 1.0, max_gain
        if median_at(hi) < target:
            gain = hi
        else:
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if median_at(mid) < target else (lo, mid)
            gain = hi
    else:
        lo, hi = 1.0 / max_gain, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if median_at(mid) < target else (lo, mid)
        gain = lo
        return _apply_masked(rgb, mask, gain, info)

    return _apply_masked(rgb, mask, max(gain, white_gain), info)


def _apply_masked(rgb: np.ndarray, mask: np.ndarray, gain: float, info: dict) -> tuple[np.ndarray, dict]:
    info["gain"] = float(gain)
    if abs(gain - 1.0) < 1e-4:
        return rgb, info
    return (rgb * _masked_gain(gain, mask)[..., None]).astype(np.float32), info


def unsharp_mask(rgb: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    """Luminance-only unsharp mask (no color fringes). amount is capped at 1."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount == 0.0:
        return rgb
    y = luminance(rgb)
    blurred = cv2.GaussianBlur(y, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    detail = y - blurred
    return np.maximum(rgb + amount * detail[..., None], 0.0).astype(np.float32)


def apply_strength(original: np.ndarray, processed: np.ndarray, strength: float) -> np.ndarray:
    s = float(np.clip(strength, 0.0, 1.0))
    if s >= 1.0:
        return processed
    if s <= 0.0:
        return original.copy()
    return (original + s * (processed - original)).astype(np.float32)
