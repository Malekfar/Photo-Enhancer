from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.evaluation import darken
from app.pipeline import run_pipeline
from app.pipeline.auto import DEFAULT_WEIGHTS, Candidate, score_output
from app.pipeline.color import luminance
from app.service import ModelRegistry
from tests.conftest import default_params, make_settings


@pytest.fixture(scope="module")
def auto_registry() -> ModelRegistry:
    reg = ModelRegistry(
        make_settings(
            default_method="auto",
            auto_candidates=[
                "retinexformer:retinexformer_lolv2_real:pc",
                "retinexformer:retinexformer_fivek:pc",
                "zerodce:zerodce_pp:pc",
                "classical::pc",
                "classical::nopc",
            ],
            auto_preview_side=256,
        )
    )
    reg.load()
    return reg


def test_candidate_parse_and_label():
    c = Candidate.parse("retinexformer:retinexformer_fivek:nopc")
    assert (c.method, c.checkpoint, c.preserve_color) == ("retinexformer", "retinexformer_fivek", False)
    assert Candidate.parse("classical::pc").label == "classical:pc"
    with pytest.raises(ValueError):
        Candidate.parse("classical")


def _score(ref, out):
    return score_output(ref, out, target_low=0.40, weights=DEFAULT_WEIGHTS, iqa=None)


def _natural(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((192, 256, 3)).astype(np.float32), (0, 0), 6)
    base = (base - base.min()) / (base.max() - base.min())
    return np.clip(0.1 + 0.75 * base + 0.03 * rng.standard_normal(base.shape), 0, 1).astype(np.float32)


def test_score_prefers_well_exposed_over_dark_or_blown():
    ref = _natural()
    dark = darken(ref, gamma=2.2, exposure=0.1, peak=1e6, read_noise=0.0, seed=0)
    good = _score(dark, ref)["score"]
    too_dark = _score(dark, np.clip(ref * 0.3, 0, 1))["score"]
    blown = _score(dark, np.clip(ref * 2.5, 0, 1))["score"]
    assert good < too_dark
    assert good < blown


def test_score_penalises_color_shift():
    ref = _natural(1)
    shifted = np.clip(ref * np.array([1.25, 1.0, 0.75], dtype=np.float32), 0, 1)
    assert _score(ref, ref)["score"] < _score(ref, shifted)["score"]


def test_auto_resolves_every_option(auto_registry, dark_scene):
    params = default_params(auto_registry.settings, method="auto", preserve_color=None)
    resolved, enhancer, info = auto_registry.resolve_params(dark_scene, params)
    assert resolved.method != "auto" and resolved.preserve_color in (True, False)
    assert enhancer.method == resolved.method
    assert info["chosen"] == info["candidates"][0]["candidate"]
    scores = [c["score"] for c in info["candidates"]]
    assert scores == sorted(scores)
    res = run_pipeline(dark_scene, resolved, auto_registry.pipeline_config(), enhancer, auto_registry.denoiser())
    # white-background scene: the background is lifted toward white, so the median is high
    assert float(np.median(luminance(res.image))) >= 0.30
    assert res.info["auto_exposure"]["white_gain"] >= 1.0


def test_auto_strength_is_full_for_dark_and_chosen_from_options(auto_registry, dark_scene):
    params = default_params(auto_registry.settings, method="auto", preserve_color=None, strength=None)
    resolved, _, info = auto_registry.resolve_params(dark_scene, params)
    assert resolved.strength in auto_registry.settings.auto_strengths
    assert info["strength"]["chosen"] == resolved.strength
    assert resolved.strength >= 0.85  # a dark input needs (nearly) the full effect


def test_explicit_method_with_auto_strength(auto_registry, dark_scene):
    params = default_params(auto_registry.settings, method="classical", preserve_color=True, strength=None)
    resolved, enhancer, info = auto_registry.resolve_params(dark_scene, params)
    assert resolved.method == "classical" and enhancer.method == "classical"
    assert resolved.strength in auto_registry.settings.auto_strengths
    assert info["chosen"] == "classical:pc"


def test_auto_respects_explicit_preserve_color(auto_registry, dark_scene):
    params = default_params(auto_registry.settings, method="auto", preserve_color=False)
    resolved, _, info = auto_registry.resolve_params(dark_scene, params)
    assert resolved.preserve_color is False
    assert all(c["candidate"].endswith(":nopc") for c in info["candidates"])


def test_explicit_method_uses_settings_default_for_preserve_color(auto_registry, dark_scene):
    params = default_params(auto_registry.settings, method="classical", preserve_color=None)
    resolved, _, info = auto_registry.resolve_params(dark_scene, params)
    assert info is None
    assert resolved.preserve_color is auto_registry.settings.preserve_color


def test_run_pipeline_rejects_unresolved_params(auto_registry, dark_scene):
    with pytest.raises(ValueError):
        run_pipeline(
            dark_scene,
            default_params(auto_registry.settings, method="auto"),
            auto_registry.pipeline_config(),
            auto_registry.enhancer("classical", None),
            None,
        )
