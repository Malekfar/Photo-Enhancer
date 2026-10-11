"""Quality regression: synthetic darkening -> pipeline -> PSNR/SSIM/LPIPS vs reference.

Run with `pytest -s tests/test_quality.py` to see the per-backend report.
Set LOL_V2_DIR=/path/to/LOL-v2/Real_captured to also evaluate on the LOL-v2 test set.
"""
from __future__ import annotations

import os
import statistics
from pathlib import Path

import numpy as np
import pytest

from app.evaluation import DEGRADATIONS, LPIPS, darken, load_lol_v2_pairs, make_product_scene, psnr, reference_images, ssim
from app.pipeline import run_pipeline
from app.pipeline.highlight import clipped_fraction
from tests.conftest import default_params, enhancer_cases


@pytest.fixture(scope="module")
def pairs():
    refs = reference_images(max_side=384)
    return [
        (f"{n}/{d.label}", darken(r, gamma=d.gamma, exposure=d.exposure, peak=d.peak, read_noise=d.read_noise, seed=i), r)
        for n, r in refs.items()
        for i, d in enumerate(DEGRADATIONS)
    ]


@pytest.fixture(scope="module")
def lpips(registry):
    m = LPIPS(registry.device)
    return m if m.available else None


def _report(label: str, rows: dict) -> None:
    def f(v):
        return "n/a" if v is None else f"{v:.4f}"

    print(
        f"\n[quality] {label:<42} PSNR {rows['psnr']:.2f} (dark {rows['psnr_dark']:.2f})"
        f" | SSIM {rows['ssim']:.4f} (dark {rows['ssim_dark']:.4f})"
        f" | LPIPS {f(rows['lpips'])} (dark {f(rows['lpips_dark'])})"
        f" | max Δclip {rows['clip_delta']*100:.3f}%"
    )


def _evaluate(registry, settings, enh, pairs, lpips, preserve_color=True):
    p = default_params(settings, method=enh.method, checkpoint=enh.checkpoint, preserve_color=preserve_color)
    acc = {k: [] for k in ("psnr", "psnr_dark", "ssim", "ssim_dark", "lpips", "lpips_dark", "clip_delta")}
    for _, low, ref in pairs:
        out = run_pipeline(low, p, registry.pipeline_config(), enh, registry.denoiser()).image
        acc["psnr"].append(psnr(out, ref))
        acc["psnr_dark"].append(psnr(low, ref))
        acc["ssim"].append(ssim(out, ref))
        acc["ssim_dark"].append(ssim(low, ref))
        if lpips is not None:
            acc["lpips"].append(lpips(out, ref))
            acc["lpips_dark"].append(lpips(low, ref))
        # Highlight protection must hold for every image, not just on average.
        delta = clipped_fraction(out) - clipped_fraction(low)
        assert delta <= settings.clip_tolerance + 1e-9, f"clip budget exceeded by {delta:.4%}"
        acc["clip_delta"].append(delta)
    rows = {k: (statistics.fmean(v) if v else None) for k, v in acc.items()}
    rows["clip_delta"] = max(acc["clip_delta"])
    return rows


@pytest.mark.parametrize("method,checkpoint", enhancer_cases())
def test_synthetic_quality(registry, settings, get_enhancer, pairs, lpips, method, checkpoint):
    enh = get_enhancer(method, checkpoint)
    rows = _evaluate(registry, settings, enh, pairs, lpips)
    _report(f"{method}:{checkpoint}", rows)
    assert rows["psnr"] > rows["psnr_dark"] + 1.0
    assert rows["ssim"] > rows["ssim_dark"]
    if rows["lpips"] is not None:
        assert rows["lpips"] < rows["lpips_dark"]


@pytest.mark.parametrize("method,checkpoint", enhancer_cases())
def test_well_exposed_white_background_stays_white(registry, settings, get_enhancer, method, checkpoint):
    """A correctly exposed product shot must keep its pure-white background."""
    enh = get_enhancer(method, checkpoint)
    ref = make_product_scene(384, 512, seed=2)
    white = ref.min(axis=-1) >= 0.99
    assert white.mean() > 0.05
    p = default_params(settings, method=method, checkpoint=checkpoint)
    out = run_pipeline(ref, p, registry.pipeline_config(), enh, registry.denoiser()).image
    assert (out[white].min(axis=-1) >= 0.99).mean() > 0.95
    assert clipped_fraction(out) <= clipped_fraction(ref) + settings.clip_tolerance + 1e-9


def test_auto_on_very_dark_images(registry, settings, lpips):
    """Very dark input (median ~0.02): auto mode must restore exposure and beat the input."""
    d = DEGRADATIONS[-1]
    refs = reference_images(max_side=384)
    acc = {"psnr": [], "psnr_dark": [], "ssim": [], "ssim_dark": []}
    for i, (name, ref) in enumerate(refs.items()):
        low = darken(ref, gamma=d.gamma, exposure=d.exposure, peak=d.peak, read_noise=d.read_noise, seed=i)
        params, enh, info = registry.resolve_params(low, default_params(settings, method="auto", preserve_color=None))
        out = run_pipeline(low, params, registry.pipeline_config(), enh, registry.denoiser()).image
        med = float(np.median(out @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
        print(f"\n[very-dark] {name:<12} chose {info['chosen']:<44} median {med:.3f} PSNR {psnr(out, ref):.2f}")
        assert med > 0.25, f"{name}: output still dark (median {med:.3f})"
        assert clipped_fraction(out) - clipped_fraction(low) <= settings.clip_tolerance + 1e-9
        acc["psnr"].append(psnr(out, ref))
        acc["psnr_dark"].append(psnr(low, ref))
        acc["ssim"].append(ssim(out, ref))
        acc["ssim_dark"].append(ssim(low, ref))
    assert statistics.fmean(acc["psnr"]) > statistics.fmean(acc["psnr_dark"]) + 3.0
    assert statistics.fmean(acc["ssim"]) > statistics.fmean(acc["ssim_dark"])


@pytest.mark.skipif(not os.environ.get("LOL_V2_DIR"), reason="set LOL_V2_DIR to evaluate on LOL-v2")
@pytest.mark.parametrize("method,checkpoint", enhancer_cases())
def test_lol_v2_quality(registry, settings, get_enhancer, lpips, method, checkpoint):
    enh = get_enhancer(method, checkpoint)
    lol = load_lol_v2_pairs(Path(os.environ["LOL_V2_DIR"]))
    limit = int(os.environ.get("LOL_V2_LIMIT", "0"))
    if limit:
        lol = lol[:limit]
    assert lol, "no LOL-v2 pairs found"
    rows = _evaluate(registry, settings, enh, lol, lpips)
    _report(f"LOL-v2 {method}:{checkpoint}", rows)
    assert rows["psnr"] > rows["psnr_dark"]
    assert np.isfinite(rows["psnr"])
