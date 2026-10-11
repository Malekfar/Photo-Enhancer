"""Highlight protection: smooth highlight mask, blending, soft-knee roll-off, clip guard."""
from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.color import CLIP_LEVEL, guided_filter, luminance, smoothstep, subsample, value_channel


def highlight_mask(y_orig: np.ndarray, t_low: float, t_high: float, radius: int, eps: float) -> np.ndarray:
    """m = smoothstep(t_low, t_high, Y), smoothed edge-aware with Y as the guide."""
    m = smoothstep(t_low, t_high, y_orig)
    return np.clip(guided_filter(y_orig, m, radius, eps), 0.0, 1.0)


def deep_shadow_mask(
    y_orig: np.ndarray,
    *,
    max_level: float,
    rel_level: float,
    min_area: float,
    radius: int,
    eps: float,
    max_texture: float = 0.0035,
) -> np.ndarray:
    """Soft mask of intentionally black regions (night sky, black backdrop) to keep as-is.

    Candidates are flat pixels much darker than the scene content:
    Y < min(max_level, rel_level * median(content)). A connected candidate region is
    protected only if it alone covers `min_area` of the frame and its mean texture is below
    `max_texture`. In a uniformly underexposed photo the dark areas are not darker than the
    content, so nothing is protected there.
    """
    h, w = y_orig.shape
    s = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(y_orig, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    texture = np.abs(small - cv2.GaussianBlur(small, (0, 0), 2.0))
    texture = cv2.GaussianBlur(texture, (0, 0), 2.0)
    flat_dark = (texture < 0.012) & (small < max_level)
    content = small[~flat_dark]
    if content.size < 0.05 * small.size:
        return np.zeros_like(y_orig)
    thr = min(max_level, rel_level * float(np.median(content)))
    cand = ((small < thr) & (texture < 0.012)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Region-level decision: only a single large, almost texture-free region counts (a sky
    # or backdrop). Scattered dark patches of real surfaces (dark wood, shadows) must not be
    # protected piecewise, or they turn into black blotches.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    m = np.zeros_like(cand)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area * cand.size:
            continue
        region = labels == i
        if float(texture[region].mean()) < max_texture:
            m[region] = 1
    if not m.any():
        return np.zeros_like(y_orig)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(guided_filter(y_orig, m, radius, eps), 0.0, 1.0)


def blend_highlights(original: np.ndarray, enhanced: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None]
    return (m * original + (1.0 - m) * enhanced).astype(np.float32)


def luminance_gain_transfer(
    base: np.ndarray, enhanced: np.ndarray, max_gain: float, radius: int = 4, eps: float = 1e-4
) -> np.ndarray:
    """Apply the enhancement as a pure luminance gain on `base` (no hue/saturation shift).

    A per-pixel ratio is unstable in near-black, noisy pixels, so the gain is the ratio of
    the two luminances after edge-aware smoothing (guided by the base luminance), which
    keeps the full gain in deep shadows without amplifying per-pixel noise in the ratio.
    """
    y_base = luminance(base)
    y_enh = luminance(enhanced)
    e = 2e-4
    num = np.maximum(guided_filter(y_base, y_enh, radius, eps), 0.0)
    den = np.maximum(guided_filter(y_base, y_base, radius, eps), 0.0)
    gain = np.clip((num + e) / (den + e), 0.0, max_gain)
    return (base * gain[..., None]).astype(np.float32)


def shoulder(v: np.ndarray | float, knee: float, white: float) -> np.ndarray | float:
    """Extended-Reinhard shoulder: identity below `knee`, maps `white` exactly to 1.0.

    C1-continuous at the knee (slope 1), strictly increasing, identity when white == 1.
    """
    white = max(float(white), 1.0)
    span = 1.0 - knee
    w = (white - knee) / span
    t = np.maximum((np.asarray(v, dtype=np.float32) - knee) / span, 0.0)
    rolled = knee + span * t * (1.0 + t / (w * w)) / (1.0 + t)
    out = np.where(np.asarray(v) > knee, np.minimum(rolled, 1.0), v)
    return out.astype(np.float32) if isinstance(out, np.ndarray) else float(out)


def soft_knee(rgb: np.ndarray, knee: float, white: float | None = None) -> tuple[np.ndarray, float]:
    """Roll off highlights on max(R,G,B), scaling RGB by the same ratio (hue-preserving).

    white defaults to the image maximum, so nothing is hard-clipped and an image that is
    already within [0, 1] is returned unchanged.
    """
    v = value_channel(rgb)
    white = max(1.0, float(v.max()) if white is None else float(white))
    if white <= 1.0:
        return np.clip(rgb, 0.0, 1.0).astype(np.float32), 1.0
    v_new = shoulder(v, knee, white)
    ratio = np.where(v > 1e-6, v_new / np.maximum(v, 1e-6), 1.0)
    return np.clip(rgb * ratio[..., None], 0.0, 1.0).astype(np.float32), white


def clipped_fraction(rgb: np.ndarray) -> float:
    return float(np.mean(value_channel(rgb) > CLIP_LEVEL))


def _fraction_after(v: np.ndarray, m: np.ndarray, knee: float, white: float) -> float:
    out_v = m * np.minimum(v, 1.0) + (1.0 - m) * shoulder(v, knee, white)
    return float(np.mean(out_v > CLIP_LEVEL))


def guard_highlights(
    rgb: np.ndarray,
    mask: np.ndarray,
    knee: float,
    allowed_fraction: float,
) -> tuple[np.ndarray, dict]:
    """Final highlight guard.

    Applies the soft-knee to the non-protected region only (mask m = 1 keeps protected
    original highlights such as white backgrounds untouched), choosing the white point so
    that the share of pixels with max(R,G,B) > 0.99 does not exceed `allowed_fraction`.
    """
    v = value_channel(rgb)
    white0 = max(1.0, float(v.max()))
    info = {"white_point": white0, "prescale": 1.0}

    v_s, m_s = subsample(v, 2_000_000), subsample(mask, 2_000_000)
    white = white0
    if _fraction_after(v_s, m_s, knee, white) > allowed_fraction or _fraction_after(v, mask, knee, white) > allowed_fraction:
        # Fraction is monotonically non-increasing in the white point: bisection.
        lo, hi = white0, white0 * 2.0
        while _fraction_after(v_s, m_s, knee, hi) > allowed_fraction and hi < white0 * 4096:
            hi *= 2.0
        if _fraction_after(v_s, m_s, knee, hi) > allowed_fraction:
            # Even an infinite shoulder cannot bring these values under 0.99: the
            # unprotected region is globally too bright. Pre-scale it down.
            q = float(np.quantile(v_s[m_s < 0.5], 1.0 - min(allowed_fraction, 1.0))) if np.any(m_s < 0.5) else 1.0
            scale = min(1.0, 2.5 / max(q, 1e-6))
            info["prescale"] = scale
            g = 1.0 + (scale - 1.0) * (1.0 - mask)
            rgb = rgb * g[..., None]
            v = value_channel(rgb)
            v_s = subsample(v, 2_000_000)
            lo, hi = max(1.0, float(v.max())), max(1.0, float(v.max())) * 4096
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _fraction_after(v_s, m_s, knee, mid) > allowed_fraction:
                lo = mid
            else:
                hi = mid
        white = hi
        # Verify on full resolution (subsampling may under-estimate); widen if needed.
        for _ in range(20):
            if _fraction_after(v, mask, knee, white) <= allowed_fraction:
                break
            white *= 1.1
    info["white_point"] = float(white)

    if white <= 1.0:
        return np.clip(rgb, 0.0, 1.0).astype(np.float32), info
    v_new = shoulder(v, knee, white)
    ratio = np.where(v > 1e-6, v_new / np.maximum(v, 1e-6), 1.0)
    rolled = rgb * ratio[..., None]
    m = mask[..., None]
    out = m * np.clip(rgb, 0.0, 1.0) + (1.0 - m) * rolled
    return np.clip(out, 0.0, 1.0).astype(np.float32), info
