"""Automatic per-image choice of backend, checkpoint, color mode and strength (no human in the loop).

Every candidate configuration is run on a downscaled preview of the (denoised) input and
scored without a reference image. The score rewards a correctly exposed, detailed result and
penalises structural change, color shift, clipping, crushed shadows and unnatural statistics.
The winner is then run once at full resolution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import cv2
import numpy as np

from app.pipeline.adjust import apply_strength
from app.pipeline.color import CLIP_LEVEL, downscale_max_side, fast_median, luminance, value_channel
from app.pipeline.denoise import Denoiser, estimate_noise_sigma, should_denoise
from app.pipeline.enhance import Enhancer
from app.pipeline.metrics import NoReferenceIQA
from app.pipeline.pipeline import EnhanceParams, PipelineConfig, run_pipeline

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "exposure": 1.0,  # (target_low - median) / target_low when under-exposed
    "crushed": 1.0,  # share of pixels still near black
    "clipped": 20.0,  # share of pixels > 0.99 beyond those already clipped in the input
    "structure": 3.0,  # 1 - structural correlation with the input
    "color": 5.0,  # mean chromaticity shift vs the input
    "entropy": 1.0,  # tonal information (subtracted: more is better)
    "niqe": 0.05,  # naturalness, per NIQE unit (if pyiqa is available)
}


@dataclass(frozen=True)
class Candidate:
    method: str
    checkpoint: str | None
    preserve_color: bool

    @property
    def label(self) -> str:
        name = f"{self.method}:{self.checkpoint}" if self.checkpoint else self.method
        return f"{name}:{'pc' if self.preserve_color else 'nopc'}"

    @classmethod
    def parse(cls, spec: str) -> "Candidate":
        """'method[:checkpoint]:pc|nopc', e.g. 'retinexformer:retinexformer_fivek:pc', 'classical::pc'."""
        parts = spec.split(":")
        if len(parts) != 3 or parts[2] not in ("pc", "nopc"):
            raise ValueError(f"bad auto candidate {spec!r}; expected 'method:checkpoint:pc|nopc'")
        return cls(parts[0], parts[1] or None, parts[2] == "pc")


def _structure(y_ref: np.ndarray, y_out: np.ndarray) -> float:
    """SSIM structure term (local correlation), invariant to brightness and contrast."""
    x = y_ref.astype(np.float64)
    y = y_out.astype(np.float64)
    med = max(float(np.median(x)), 1e-3)
    x = np.clip(x * (0.45 / med), 0.0, 1.0)  # same scale as a well-exposed image
    c3 = (0.03**2) / 2.0

    def blur(z: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(z, (7, 7), 1.5, borderType=cv2.BORDER_REFLECT)

    mx, my = blur(x), blur(y)
    sx = np.sqrt(np.maximum(blur(x * x) - mx * mx, 0.0))
    sy = np.sqrt(np.maximum(blur(y * y) - my * my, 0.0))
    sxy = blur(x * y) - mx * my
    return float(np.mean((sxy + c3) / (sx * sy + c3)))


def _color_shift(ref: np.ndarray, out: np.ndarray) -> float:
    """Mean chromaticity (r,g,b / sum) difference, ignoring near-black reference pixels."""
    eps = 1e-4
    c_ref = ref / (ref.sum(axis=-1, keepdims=True) + eps)
    c_out = out / (out.sum(axis=-1, keepdims=True) + eps)
    w = np.clip(luminance(ref) / max(float(np.percentile(luminance(ref), 75)), 1e-3), 0.0, 1.0)
    diff = np.abs(c_ref - c_out).sum(axis=-1)
    return float((diff * w).sum() / max(float(w.sum()), 1e-6))


def _entropy(y: np.ndarray) -> float:
    hist = np.bincount(np.clip(np.rint(y * 255), 0, 255).astype(np.int32).ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / 8.0)


def score_output(
    ref: np.ndarray,
    out: np.ndarray,
    *,
    target_low: float,
    weights: dict[str, float],
    iqa: NoReferenceIQA | None,
) -> dict:
    y_out = luminance(out)
    med = float(np.median(y_out))
    # Under-exposure only: bright-background product shots legitimately sit far above the
    # target band; over-exposure shows up in the clipping and structure terms instead.
    exposure = max(0.0, (target_low - med) / target_low)
    comp = {
        "median": med,
        "exposure": exposure,
        "crushed": float(np.mean(y_out < 0.03)),
        "clipped": max(
            0.0, float(np.mean(value_channel(out) > CLIP_LEVEL)) - float(np.mean(value_channel(ref) > CLIP_LEVEL))
        ),
        "structure": _structure(luminance(ref), y_out),
        "color": _color_shift(ref, out),
        "entropy": _entropy(y_out),
        "niqe": None,
    }
    if iqa is not None and weights.get("niqe", 0) > 0:
        comp["niqe"] = iqa.score(out).get("niqe")
    score = (
        weights["exposure"] * comp["exposure"]
        + weights["crushed"] * comp["crushed"]
        + weights["clipped"] * comp["clipped"]
        + weights["structure"] * (1.0 - comp["structure"])
        + weights["color"] * comp["color"]
        - weights["entropy"] * comp["entropy"]
    )
    if comp["niqe"] is not None:
        score += weights["niqe"] * comp["niqe"]
    comp["score"] = score
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in comp.items()}


def select_candidate(
    rgb: np.ndarray,
    candidates: list[tuple[Candidate, Enhancer]],
    params: EnhanceParams,
    cfg: PipelineConfig,
    denoiser: Denoiser | None,
    iqa: NoReferenceIQA | None,
    *,
    weights: dict[str, float],
    preview_side: int,
    strengths: list[float] | None = None,
) -> tuple[Candidate, float | None, dict]:
    """Pick the best candidate; if `params.strength` is None also pick the strength.

    Strength is chosen by blending the winner's preview output with the preview input at
    each of `strengths` and scoring the blends the same way: a dark input needs the full
    effect, an input that is already well exposed gets a light touch.
    """
    if not candidates:
        raise ValueError("no auto candidates available")
    preview = downscale_max_side(rgb, preview_side)
    sigma = estimate_noise_sigma(preview)
    apply, _ = should_denoise(
        params.denoise,
        sigma,
        fast_median(luminance(preview)),
        dark_threshold=cfg.dark_threshold,
        noise_threshold=cfg.noise_threshold,
        target_median=0.5 * (cfg.ae_target_low + cfg.ae_target_high),
    )
    base = preview
    if apply and denoiser is not None and params.denoise_strength > 0:
        base = denoiser.denoise(preview, params.denoise_strength)

    run_strength = 1.0 if params.strength is None else params.strength
    scored: list[tuple[float, Candidate, dict, np.ndarray]] = []
    for cand, enh in candidates:
        p = replace(
            params,
            method=cand.method,
            checkpoint=cand.checkpoint,
            preserve_color=cand.preserve_color,
            strength=run_strength,
            denoise="off",
        )
        try:
            out = run_pipeline(base, p, cfg, enh, None).image
        except Exception as exc:  # noqa: BLE001 - one failing backend must not fail the request
            log.warning("auto candidate failed", extra={"candidate": cand.label, "error": str(exc)})
            continue
        sc = score_output(base, out, target_low=cfg.ae_target_low, weights=weights, iqa=iqa)
        scored.append((sc["score"], cand, sc, out))
    if not scored:
        raise RuntimeError("all auto candidates failed")
    scored.sort(key=lambda t: t[0])
    best, best_out = scored[0][1], scored[0][3]
    info: dict = {
        "chosen": best.label,
        "preview_size": [base.shape[1], base.shape[0]],
        "candidates": [{"candidate": c.label, **sc} for _, c, sc, _ in scored],
    }

    strength = None
    if params.strength is None:
        options = []
        for s in sorted(strengths or [1.0]):
            blend = apply_strength(preview, best_out, s)
            sc = score_output(base, blend, target_low=cfg.ae_target_low, weights=weights, iqa=iqa)
            options.append((sc["score"], s))
        strength = min(options)[1]
        info["strength"] = {"chosen": strength, "scores": {str(s): round(v, 5) for v, s in options}}
    return best, strength, info
