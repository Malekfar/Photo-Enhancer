"""Pipeline orchestration. Stages are independent modules and swappable via the registry."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from app.pipeline.adjust import apply_strength, auto_exposure, pre_exposure, unsharp_mask
from app.pipeline.color import fast_median, luminance, subsample, value_channel
from app.pipeline.denoise import Denoiser, estimate_noise_sigma, should_denoise, suppress_amplified_chroma
from app.pipeline.enhance import Enhancer
from app.pipeline.highlight import (
    blend_highlights,
    clipped_fraction,
    deep_shadow_mask,
    guard_highlights,
    highlight_mask,
    luminance_gain_transfer,
    soft_knee,
)


@dataclass
class EnhanceParams:
    method: Literal["auto", "retinexformer", "zerodce", "classical"]
    checkpoint: str | None
    strength: float | None  # None = decided automatically
    denoise: Literal["auto", "on", "off"]
    denoise_strength: float
    highlight_low: float
    highlight_high: float
    preserve_color: bool | None  # None = decided automatically
    sharpen: bool
    output_format: Literal["jpeg", "png", "webp"]
    quality: int
    strip_exif: bool
    iqa: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineConfig:
    """Tuning constants not exposed per request (set via environment)."""

    knee: float
    clip_tolerance: float
    max_gain: float
    mask_radius: int
    mask_eps: float
    ae_target_low: float
    ae_target_high: float
    ae_max_gain: float
    ae_allow_darken: bool
    sharpen_amount: float
    sharpen_sigma: float
    dark_threshold: float
    noise_threshold: float
    pre_gain_targets: dict[str, float] = field(default_factory=dict)
    pre_gain_max: float = 1024.0
    ae_white_target: float | None = 0.93
    ae_bg_min_fraction: float = 0.2
    ae_white_max_input: float = 0.7  # only for low-light input: original p99 of max(R,G,B) below this
    shadow_protect: bool = True
    shadow_max_level: float = 0.08
    shadow_rel_level: float = 0.3
    shadow_min_area: float = 0.10
    shadow_max_texture: float = 0.0035
    chroma_suppress: bool = True
    chroma_gain_low: float = 2.0
    chroma_gain_high: float = 5.0
    chroma_radius: int = 8

    def pre_gain_target(self, enhancer: Enhancer) -> float | None:
        if enhancer.checkpoint and enhancer.checkpoint in self.pre_gain_targets:
            return self.pre_gain_targets[enhancer.checkpoint]
        return self.pre_gain_targets.get(enhancer.method)


@dataclass
class PipelineResult:
    image: np.ndarray
    info: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)


class _Timer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def __call__(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()

            def __exit__(self_inner, *exc):
                timer.timings[name] = round((time.perf_counter() - self_inner.t0) * 1000.0, 1)

        return _Ctx()


def run_pipeline(
    rgb: np.ndarray,
    params: EnhanceParams,
    cfg: PipelineConfig,
    enhancer: Enhancer,
    denoiser: Denoiser | None,
) -> PipelineResult:
    """original -> [denoise] -> [linear pre-exposure] -> enhance -> [luma-gain] -> roll-off -> highlight blend
    -> auto-exposure -> [sharpen] -> strength -> highlight guard."""
    if not 0.0 <= params.highlight_low < params.highlight_high <= 1.0:
        raise ValueError("require 0 <= highlight_low < highlight_high <= 1")
    if params.method == "auto" or params.preserve_color is None or params.strength is None:
        raise ValueError("method, preserve_color and strength must be resolved before run_pipeline")
    timer = _Timer()
    info: dict = {}
    original = rgb.astype(np.float32, copy=False)
    allowed = clipped_fraction(original) + cfg.clip_tolerance

    # 1. denoise
    with timer("denoise"):
        y0 = luminance(original)
        med0 = fast_median(y0)
        sigma = estimate_noise_sigma(original)
        apply, projected = should_denoise(
            params.denoise,
            sigma,
            med0,
            dark_threshold=cfg.dark_threshold,
            noise_threshold=cfg.noise_threshold,
            target_median=0.5 * (cfg.ae_target_low + cfg.ae_target_high),
        )
        base = original
        if apply and denoiser is not None and params.denoise_strength > 0:
            base = denoiser.denoise(original, params.denoise_strength)
        info["denoise"] = {
            "applied": bool(apply and denoiser is not None and params.denoise_strength > 0),
            "backend": denoiser.name if denoiser is not None else None,
            "noise_sigma": round(sigma, 5),
            "projected_noise": round(projected, 5),
        }
        if cfg.shadow_protect:
            shadow = deep_shadow_mask(
                y0,
                max_level=cfg.shadow_max_level,
                rel_level=cfg.shadow_rel_level,
                min_area=cfg.shadow_min_area,
                max_texture=cfg.shadow_max_texture,
                radius=cfg.mask_radius,
                eps=cfg.mask_eps,
            )
        else:
            shadow = np.zeros_like(y0)

    # 2. exposure pre-normalisation (very dark input -> model's trained brightness range)
    with timer("pre_exposure"):
        model_input, pre_info = pre_exposure(
            base, cfg.pre_gain_target(enhancer), cfg.pre_gain_max, cfg.knee, exclude=shadow
        )
        info["pre_exposure"] = pre_info

    # 3. enhance
    with timer("enhance"):
        res = enhancer.enhance(model_input)
        enhanced = res.image
        info["enhancer"] = enhancer.describe() | res.info

    # 4. highlight protection
    with timer("highlight"):
        if params.preserve_color:
            enhanced = luminance_gain_transfer(base, enhanced, cfg.max_gain)
        enhanced, white = soft_knee(enhanced, cfg.knee)
        mask = highlight_mask(y0, params.highlight_low, params.highlight_high, cfg.mask_radius, cfg.mask_eps)
        out = blend_highlights(base, enhanced, mask)
        out = blend_highlights(base, out, shadow)  # keep intentional blacks (night sky) black
        info["highlight"] = {
            "enhanced_white_point": round(white, 4),
            "protected_pct": round(float(np.mean(mask > 0.5)) * 100.0, 3),
            "shadow_protected_pct": round(float(np.mean(shadow > 0.5)) * 100.0, 3),
        }

    # 5. final adjustments
    with timer("adjust"):
        white_target = None
        if cfg.ae_white_target is not None:
            p_in = float(np.percentile(subsample(value_channel(original), 500_000), 99.0))
            if p_in < cfg.ae_white_max_input:
                white_target = cfg.ae_white_target
        out, ae_info = auto_exposure(
            out,
            np.maximum(mask, shadow),
            cfg.ae_target_low,
            cfg.ae_target_high,
            cfg.ae_max_gain,
            cfg.ae_allow_darken,
            white_target=white_target,
            bg_min_fraction=cfg.ae_bg_min_fraction,
            exclude=shadow,
        )
        info["auto_exposure"] = {k: round(v, 4) for k, v in ae_info.items()}
        if params.sharpen:
            out = unsharp_mask(out, cfg.sharpen_amount, cfg.sharpen_sigma)
        out = apply_strength(original, out, params.strength)
        if cfg.chroma_suppress:
            radius = max(cfg.chroma_radius, round(max(out.shape[:2]) / 128))
            out, chroma_pct = suppress_amplified_chroma(
                base, out, gain_low=cfg.chroma_gain_low, gain_high=cfg.chroma_gain_high, radius=radius, eps=1e-3
            )
            info["chroma_suppress"] = {"affected_pct": round(chroma_pct * 100.0, 3), "radius": radius}

    # 6. guard: soft roll-off (no hard clip) under the clipped-pixel budget
    with timer("guard"):
        out, guard_info = guard_highlights(out, mask, cfg.knee, allowed)
        info["highlight"].update(
            final_white_point=round(guard_info["white_point"], 4),
            prescale=round(guard_info["prescale"], 4),
            allowed_clipped_pct=round(allowed * 100.0, 4),
        )

    return PipelineResult(image=out, info=info, timings_ms=timer.timings)
