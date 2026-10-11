from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.evaluation import make_product_scene
from app.pipeline import EnhanceParams
from app.service import ModelRegistry

WEIGHTS_DIR = Path(os.environ.get("LLE_WEIGHTS_DIR", "/weights"))


def make_settings(**overrides) -> Settings:
    base = dict(
        weights_dir=WEIGHTS_DIR,
        iqa_enabled=False,
        log_json=False,
        log_level="WARNING",
        max_concurrency=2,
        default_method="classical",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return make_settings()


@pytest.fixture(scope="session")
def registry(settings: Settings) -> ModelRegistry:
    reg = ModelRegistry(settings)
    reg.load()
    return reg


@pytest.fixture
def get_enhancer(registry: ModelRegistry):
    def _get(method: str, checkpoint: str | None = None):
        cks = registry.enhancers.get(method, {})
        if not cks or (checkpoint is not None and checkpoint not in cks):
            pytest.skip(f"{method}:{checkpoint} not loaded ({registry.failures})")
        return registry.enhancer(method, checkpoint)

    return _get


def enhancer_cases() -> list[tuple[str, str | None]]:
    return [
        ("retinexformer", "retinexformer_fivek"),
        ("retinexformer", "retinexformer_lolv2_real"),
        ("zerodce", None),
        ("classical", None),
    ]


def default_params(settings: Settings, **overrides) -> EnhanceParams:
    p = dict(
        method="classical",
        checkpoint=None,
        strength=1.0,
        denoise="auto",
        denoise_strength=1.0,
        highlight_low=settings.highlight_low,
        highlight_high=settings.highlight_high,
        preserve_color=True,
        sharpen=False,
        output_format="png",
        quality=95,
        strip_exif=True,
        iqa=False,
    )
    p.update(overrides)
    return EnhanceParams(**p)


@pytest.fixture(scope="session")
def scene() -> np.ndarray:
    return make_product_scene(256, 384, seed=0)


@pytest.fixture(scope="session")
def dark_scene(scene: np.ndarray) -> np.ndarray:
    from app.evaluation import darken

    return darken(scene, gamma=3.0, exposure=0.2, peak=300.0, read_noise=0.006, seed=0)
