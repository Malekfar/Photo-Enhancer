from __future__ import annotations

import numpy as np
import pytest

from app.pipeline import run_pipeline
from app.pipeline.highlight import clipped_fraction
from tests.conftest import default_params


def test_pipeline_output_contract(registry, settings, dark_scene):
    p = default_params(settings)
    res = run_pipeline(dark_scene, p, registry.pipeline_config(), registry.enhancer("classical", None), registry.denoiser())
    assert res.image.shape == dark_scene.shape
    assert res.image.dtype == np.float32
    assert res.image.min() >= 0.0 and res.image.max() <= 1.0
    for stage in ("denoise", "enhance", "highlight", "adjust", "guard"):
        assert stage in res.timings_ms
    assert res.info["denoise"]["applied"] in (True, False)


def test_strength_zero_returns_input(registry, settings, dark_scene):
    p = default_params(settings, strength=0.0)
    res = run_pipeline(dark_scene, p, registry.pipeline_config(), registry.enhancer("classical", None), registry.denoiser())
    np.testing.assert_allclose(res.image, dark_scene, atol=1e-6)


def test_strength_is_monotone(registry, settings, dark_scene):
    meds = []
    for s in (0.0, 0.5, 1.0):
        p = default_params(settings, strength=s, denoise="off")
        res = run_pipeline(dark_scene, p, registry.pipeline_config(), registry.enhancer("classical", None), None)
        meds.append(float(np.median(res.image)))
    assert meds[0] < meds[1] < meds[2]


def test_invalid_highlight_range(registry, settings, dark_scene):
    p = default_params(settings, highlight_low=0.9, highlight_high=0.6)
    with pytest.raises(ValueError):
        run_pipeline(dark_scene, p, registry.pipeline_config(), registry.enhancer("classical", None), None)


def test_clip_budget_with_bright_spots(registry, settings, dark_scene):
    img = dark_scene.copy()
    img[10:30, 10:60] = 1.0  # a clipped light source in the dark scene
    p = default_params(settings, denoise="off")
    res = run_pipeline(img, p, registry.pipeline_config(), registry.enhancer("classical", None), None)
    assert clipped_fraction(res.image) <= clipped_fraction(img) + settings.clip_tolerance + 1e-9
    np.testing.assert_allclose(res.image[16:24, 25:45], 1.0, atol=1e-2)  # light source not dimmed
