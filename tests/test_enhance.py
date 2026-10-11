from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from app.pipeline.color import luminance
from app.pipeline.enhance import ClassicalEnhancer, RetinexformerEnhancer, ZeroDCEEnhancer
from tests.conftest import enhancer_cases


@pytest.mark.parametrize("method,checkpoint", enhancer_cases())
def test_enhancer_output_contract(get_enhancer, dark_scene, method, checkpoint):
    enh = get_enhancer(method, checkpoint)
    res = enh.enhance(dark_scene)
    out = res.image
    assert out.shape == dark_scene.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0
    assert np.median(luminance(out)) > np.median(luminance(dark_scene)) * 1.1


@pytest.mark.parametrize("method,checkpoint", enhancer_cases())
def test_enhancer_handles_odd_sizes(get_enhancer, dark_scene, method, checkpoint):
    enh = get_enhancer(method, checkpoint)
    crop = dark_scene[:37, :53]
    assert enh.enhance(crop).image.shape == (37, 53, 3)


def _clone_rf(enh: RetinexformerEnhancer, **kw) -> RetinexformerEnhancer:
    args = dict(
        max_model_pixels=enh.max_model_pixels,
        large_mode=enh.large_mode,
        tile=enh.tile,
        overlap=enh.overlap,
        guided_radius=enh.guided_radius,
        guided_eps=enh.guided_eps,
    )
    args.update(kw)
    return RetinexformerEnhancer(enh.model, enh.checkpoint, enh.device, **args)


def test_retinexformer_guided_mode_for_large_images(get_enhancer, dark_scene):
    enh = _clone_rf(get_enhancer("retinexformer", "retinexformer_fivek"), max_model_pixels=96 * 64, large_mode="guided")
    res = enh.enhance(dark_scene)
    assert res.info["mode"] == "guided"
    assert res.image.shape == dark_scene.shape
    assert np.isfinite(res.image).all()


def test_retinexformer_tile_mode_matches_full(get_enhancer, dark_scene):
    base = get_enhancer("retinexformer", "retinexformer_fivek")
    full = base.enhance(dark_scene).image
    tiled = _clone_rf(base, max_model_pixels=4096, large_mode="tile", tile=128, overlap=32).enhance(dark_scene)
    assert tiled.info["mode"] == "tile"
    assert tiled.image.shape == full.shape
    # Retinexformer attention is global, so tiles differ slightly; no gross deviation/seams.
    assert float(np.mean(np.abs(tiled.image - full))) < 0.05


def test_zerodce_downscaled_params_close_to_full(get_enhancer, dark_scene):
    enh = get_enhancer("zerodce")
    full = ZeroDCEEnhancer(enh.model, enh.checkpoint, enh.device, max_side=4096).enhance(dark_scene).image
    small = ZeroDCEEnhancer(enh.model, enh.checkpoint, enh.device, max_side=128).enhance(dark_scene).image
    assert small.shape == full.shape
    assert float(np.mean(np.abs(small - full))) < 0.08


def test_zerodce_row_upsampling_equals_full_interpolation():
    torch.manual_seed(0)
    x_r = torch.rand(1, 3, 17, 23)
    h, w = 101, 137
    full = F.interpolate(x_r, size=(h, w), mode="bilinear", align_corners=False)
    for y0, y1 in [(0, 40), (40, 77), (77, 101)]:
        part = ZeroDCEEnhancer._upsample_rows(x_r, h, w, y0, y1)
        assert torch.allclose(part, full[..., y0:y1, :], atol=1e-5)


def test_classical_is_monotone_brightening():
    enh = ClassicalEnhancer(clip_limit=2.0, grid=8, target=0.45, max_gain=8.0)
    ramp = np.tile(np.linspace(0.01, 0.3, 256, dtype=np.float32)[None, :, None], (64, 1, 3))
    out = enh.enhance(ramp).image
    assert out.shape == ramp.shape
    assert out.max() <= 1.0
    assert np.median(luminance(out)) > np.median(luminance(ramp))
    # Neutral input stays neutral (no color cast).
    assert np.abs(out[..., 0] - out[..., 2]).max() < 0.02
