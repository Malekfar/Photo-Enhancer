from __future__ import annotations

import numpy as np

from app.pipeline.adjust import apply_strength, auto_exposure, pre_exposure, unsharp_mask
from app.pipeline.color import linear_to_srgb, luminance, srgb_to_linear


def _img(level: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(level * (0.6 + 0.8 * rng.random((64, 64, 3))), 0, 1).astype(np.float32)


def test_auto_exposure_reaches_target():
    img = _img(0.2)
    mask = np.zeros(img.shape[:2], dtype=np.float32)
    out, info = auto_exposure(img, mask, 0.40, 0.50, max_gain=4.0, allow_darken=False)
    med = float(np.median(np.minimum(luminance(out), 1.0)))
    assert 0.39 <= med <= 0.51
    assert info["gain"] > 1.0


def test_auto_exposure_gain_capped():
    img = _img(0.02)
    mask = np.zeros(img.shape[:2], dtype=np.float32)
    _, info = auto_exposure(img, mask, 0.40, 0.50, max_gain=4.0, allow_darken=False)
    assert info["gain"] <= 4.0 + 1e-6


def test_auto_exposure_never_darkens_by_default():
    img = _img(0.8)
    mask = np.zeros(img.shape[:2], dtype=np.float32)
    out, info = auto_exposure(img, mask, 0.40, 0.50, max_gain=4.0, allow_darken=False)
    assert info["gain"] == 1.0
    np.testing.assert_array_equal(out, img)


def test_auto_exposure_can_darken_when_allowed():
    img = _img(0.8)
    mask = np.zeros(img.shape[:2], dtype=np.float32)
    out, _ = auto_exposure(img, mask, 0.40, 0.50, max_gain=4.0, allow_darken=True)
    assert float(np.median(luminance(out))) <= 0.51


def test_auto_exposure_leaves_protected_pixels():
    img = _img(0.2)
    mask = np.zeros(img.shape[:2], dtype=np.float32)
    mask[:, :8] = 1.0
    out, _ = auto_exposure(img, mask, 0.40, 0.50, max_gain=4.0, allow_darken=False)
    np.testing.assert_allclose(out[:, :8], img[:, :8])


def test_srgb_linear_roundtrip():
    x = np.linspace(0, 1, 1001, dtype=np.float32)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_pre_exposure_lifts_very_dark_input_to_target():
    img = _img(0.01)
    out, info = pre_exposure(img, target=0.2, max_gain=1024.0, knee=0.85)
    med = float(np.median(luminance(out)))
    assert 0.18 <= med <= 0.22
    assert info["gain"] > 10.0
    assert out.max() <= 1.0


def test_pre_exposure_is_a_global_gain_in_linear_light():
    img = _img(0.02)
    out, info = pre_exposure(img, target=0.1, max_gain=1024.0, knee=0.85)
    lin_in, lin_out = srgb_to_linear(img), srgb_to_linear(out)
    sel = lin_in > 1e-3
    ratio = lin_out[sel] / lin_in[sel]
    np.testing.assert_allclose(ratio, info["gain"], rtol=2e-3)


def test_pre_exposure_noop_cases():
    img = _img(0.4)
    out, info = pre_exposure(img, target=0.2, max_gain=1024.0, knee=0.85)
    assert info["gain"] == 1.0 and out is img
    out, info = pre_exposure(_img(0.01), target=None, max_gain=1024.0, knee=0.85)
    assert info["gain"] == 1.0


def test_pre_exposure_respects_cap():
    _, info = pre_exposure(_img(0.005), target=0.4, max_gain=8.0, knee=0.85)
    assert info["gain"] == 8.0


def test_strength_blend():
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = np.ones((4, 4, 3), dtype=np.float32)
    np.testing.assert_array_equal(apply_strength(a, b, 0.0), a)
    np.testing.assert_array_equal(apply_strength(a, b, 1.0), b)
    np.testing.assert_allclose(apply_strength(a, b, 0.25), 0.25)


def test_unsharp_identity_and_flat():
    img = _img(0.5)
    np.testing.assert_array_equal(unsharp_mask(img, 0.0, 1.0), img)
    flat = np.full((32, 32, 3), 0.4, dtype=np.float32)
    np.testing.assert_allclose(unsharp_mask(flat, 0.5, 1.0), flat, atol=1e-6)


def test_unsharp_increases_edge_contrast_mildly():
    img = np.full((32, 32, 3), 0.3, dtype=np.float32)
    img[:, 16:] = 0.6
    out = unsharp_mask(img, 0.3, 1.0)
    assert out[:, 16].mean() > 0.6 and out[:, 15].mean() < 0.3
    assert out.max() < 0.6 + 0.3 * 0.3 + 1e-3  # bounded overshoot
    assert np.abs(out[:, 0] - img[:, 0]).max() < 1e-5  # flat areas untouched
