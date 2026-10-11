from __future__ import annotations

import numpy as np

from app.pipeline.color import luminance, smoothstep
from app.pipeline.highlight import (
    blend_highlights,
    clipped_fraction,
    guard_highlights,
    highlight_mask,
    luminance_gain_transfer,
    shoulder,
    soft_knee,
)


def test_smoothstep_endpoints_and_monotonic():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    s = smoothstep(0.6, 0.9, x)
    assert s[x <= 0.6].max() == 0.0
    assert s[x >= 0.9].min() == 1.0
    assert np.all(np.diff(s) >= 0)


def test_highlight_mask_regions():
    y = np.full((64, 128), 0.3, dtype=np.float32)
    y[:, 64:] = 0.97
    m = highlight_mask(y, 0.6, 0.9, radius=4, eps=1e-3)
    assert m.shape == y.shape and m.dtype == np.float32
    assert m[:, :48].max() < 0.02
    assert m[:, 80:].min() > 0.98
    assert 0.0 <= m.min() and m.max() <= 1.0


def test_blend_uses_original_in_highlights():
    orig = np.full((4, 4, 3), 0.95, dtype=np.float32)
    enh = np.full((4, 4, 3), 1.4, dtype=np.float32)
    mask = np.ones((4, 4), dtype=np.float32)
    np.testing.assert_allclose(blend_highlights(orig, enh, mask), orig)
    np.testing.assert_allclose(blend_highlights(orig, enh, 0 * mask), enh)


def test_shoulder_properties():
    v = np.linspace(0, 1.6, 1601, dtype=np.float32)
    out = shoulder(v, knee=0.85, white=1.6)
    below = v <= 0.85
    np.testing.assert_allclose(out[below], v[below])
    assert np.all(np.diff(out) >= -1e-7)
    assert abs(out[-1] - 1.0) < 1e-5
    assert out.max() <= 1.0
    ident = shoulder(np.linspace(0, 1, 11, dtype=np.float32), knee=0.85, white=1.0)
    np.testing.assert_allclose(ident, np.linspace(0, 1, 11), atol=1e-6)


def test_soft_knee_identity_in_range_and_no_hard_clip():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 3)).astype(np.float32)
    out, white = soft_knee(img, 0.85)
    assert white == 1.0
    np.testing.assert_allclose(out, img)

    hot = img.copy()
    hot[:4] *= 1.6
    out, white = soft_knee(hot, 0.85)
    assert white > 1.0 and out.max() <= 1.0
    low = hot.max(axis=-1) <= 0.85
    np.testing.assert_allclose(out[low], hot[low], atol=1e-6)
    # Distinct overexposed values stay distinct (roll-off, not clipping).
    assert len(np.unique(np.round(out[:4].max(axis=-1), 4))) > 50


def test_guard_enforces_clip_budget():
    rng = np.random.default_rng(1)
    img = (rng.random((100, 100, 3)) * 0.5).astype(np.float32)
    img[:10] = 1.5  # 10% overexposed after enhancement
    mask = np.zeros((100, 100), dtype=np.float32)
    out, info = guard_highlights(img, mask, knee=0.85, allowed_fraction=0.002)
    assert clipped_fraction(out) <= 0.002 + 1e-9
    assert info["white_point"] > 1.5
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_guard_preserves_protected_white_background():
    img = np.full((50, 50, 3), 0.4, dtype=np.float32)
    img[:, :25] = 1.0  # white background (protected)
    img[:5, 25:] = 1.3  # overexposed product region (unprotected)
    mask = np.zeros((50, 50), dtype=np.float32)
    mask[:, :25] = 1.0
    out, _ = guard_highlights(img, mask, knee=0.85, allowed_fraction=0.5 + 0.002)
    np.testing.assert_allclose(out[:, :25], 1.0)
    assert out[:5, 25:].max() <= 0.99


def test_luminance_gain_transfer_preserves_hue():
    rng = np.random.default_rng(2)
    base = (0.05 + 0.2 * rng.random((32, 32, 3))).astype(np.float32)
    enhanced = np.clip(base * 3.0 + np.array([0.1, -0.05, 0.0], dtype=np.float32), 0, None)  # color-shifted
    out = luminance_gain_transfer(base, enhanced, max_gain=8.0)
    ratio_in = base / base.sum(-1, keepdims=True)
    ratio_out = out / out.sum(-1, keepdims=True)
    np.testing.assert_allclose(ratio_out, ratio_in, atol=1e-5)
    # gain map is smoothed, so luminance matches on average rather than per pixel
    assert abs(float(luminance(out).mean()) / float(luminance(enhanced).mean()) - 1.0) < 0.05


def test_luminance_gain_is_capped():
    base = np.full((4, 4, 3), 0.05, dtype=np.float32)
    enhanced = np.full((4, 4, 3), 1.0, dtype=np.float32)
    out = luminance_gain_transfer(base, enhanced, max_gain=4.0)
    np.testing.assert_allclose(out, 0.2, atol=1e-5)


def test_luminance_gain_transfer_keeps_near_black_colored_not_gray():
    base = np.zeros((16, 16, 3), dtype=np.float32)
    base[..., 0] = 0.006  # very dark red
    base[..., 2] = 0.002
    enhanced = base * 30.0
    out = luminance_gain_transfer(base, enhanced, max_gain=48.0)
    assert out[..., 0].mean() > 3 * out[..., 1].mean() + 0.01  # still red, not gray
    assert luminance(out).mean() > 10 * luminance(base).mean()
