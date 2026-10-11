from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.denoise import OpenCVDenoiser, estimate_noise_sigma, should_denoise


@pytest.mark.parametrize("sigma", [0.01, 0.03, 0.06])
def test_noise_estimate_on_flat_image(sigma):
    rng = np.random.default_rng(0)
    img = np.clip(0.5 + rng.normal(0, sigma, (256, 256, 1)), 0, 1).astype(np.float32).repeat(3, axis=2)
    est = estimate_noise_sigma(img)
    assert abs(est - sigma) / sigma < 0.2


def test_noise_estimate_increases_with_noise(scene):
    rng = np.random.default_rng(3)
    noisy = np.clip(scene + rng.normal(0, 0.03, scene.shape), 0, 1).astype(np.float32)
    assert estimate_noise_sigma(noisy) > estimate_noise_sigma(scene) + 0.01


def _noisy_dark(scene, seed=0):
    rng = np.random.default_rng(seed)
    clean = (scene * 0.15).astype(np.float32)
    noisy = np.clip(clean + rng.normal(0, 0.01, clean.shape), 0, 1).astype(np.float32)
    return clean, noisy


def _check_denoiser(den, scene):
    clean, noisy = _noisy_dark(scene)
    out = den.denoise(noisy, 1.0)
    assert out.shape == noisy.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    mse_in = float(np.mean((noisy - clean) ** 2))
    mse_out = float(np.mean((out - clean) ** 2))
    assert mse_out < 0.7 * mse_in
    np.testing.assert_array_equal(den.denoise(noisy, 0.0), noisy)


def test_opencv_denoiser(scene):
    _check_denoiser(OpenCVDenoiser(), scene)


def test_scunet_denoiser(registry, scene):
    if "scunet" not in registry.denoisers:
        pytest.skip(f"scunet not loaded: {registry.failures.get(registry.settings.scunet_checkpoint)}")
    _check_denoiser(registry.denoisers["scunet"], scene)


def test_should_denoise_logic():
    kw = dict(dark_threshold=0.3, noise_threshold=0.02, target_median=0.45)
    assert should_denoise("auto", 0.004, 0.05, **kw)[0] is True  # dark + noise amplified 9x
    assert should_denoise("auto", 0.004, 0.6, **kw)[0] is False  # bright image
    assert should_denoise("auto", 0.0005, 0.05, **kw)[0] is False  # clean dark image
    assert should_denoise("on", 0.0, 0.9, **kw)[0] is True
    assert should_denoise("off", 0.1, 0.01, **kw)[0] is False
