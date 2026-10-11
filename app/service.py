"""Model registry (loaded once at startup), inference runner and request processing."""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import cv2
import numpy as np
import torch

from app.config import Settings
from app.errors import BackendUnavailableError, RequestTimeoutError, WeightsError
from app.models import build_retinexformer, build_scunet, build_zerodce
from app.pipeline import EnhanceParams, PipelineConfig, run_pipeline
from app.pipeline.auto import DEFAULT_WEIGHTS, Candidate, select_candidate
from app.pipeline.denoise import Denoiser, OpenCVDenoiser, SCUNetDenoiser
from app.pipeline.enhance import ClassicalEnhancer, Enhancer, RetinexformerEnhancer, ZeroDCEEnhancer
from app.pipeline.io import decode_image, encode_image
from app.pipeline.metrics import NoReferenceIQA, luminance_stats
from app.weights import load_manifest, verify_file

log = logging.getLogger(__name__)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"LLE_DEVICE={spec} but CUDA is not available")
    return dev


@dataclass
class ModelRegistry:
    settings: Settings
    device: torch.device = torch.device("cpu")
    enhancers: dict[str, dict[str | None, Enhancer]] = field(default_factory=dict)
    denoisers: dict[str, Denoiser] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    iqa: NoReferenceIQA | None = None

    def load(self) -> None:
        s = self.settings
        self.device = resolve_device(s.device)
        if s.torch_threads:
            torch.set_num_threads(s.torch_threads)
        manifest = load_manifest(s.weights_manifest)

        def weight_path(name: str):
            if name not in manifest:
                raise WeightsError(f"'{name}' is not defined in {s.weights_manifest}")
            spec = manifest[name]
            path = s.weights_dir / spec.file
            if s.verify_weights:
                verify_file(spec, s.weights_dir, allow_unpinned=s.allow_unpinned_weights)
            elif not path.is_file():
                raise WeightsError(f"weights '{name}' missing at {path}")
            return spec, path

        if "retinexformer" in s.enabled_backends:
            for ck in s.retinexformer_checkpoints:
                try:
                    spec, path = weight_path(ck)
                    model = build_retinexformer(path, spec.arch, self.device)
                    self.enhancers.setdefault("retinexformer", {})[ck] = RetinexformerEnhancer(
                        model,
                        ck,
                        self.device,
                        max_model_pixels=s.enhance_max_model_pixels,
                        large_mode=s.large_image_mode,
                        tile=s.tile_size,
                        overlap=s.tile_overlap,
                        guided_radius=s.guided_upsample_radius,
                        guided_eps=s.guided_upsample_eps,
                    )
                    log.info("loaded retinexformer checkpoint", extra={"checkpoint": ck})
                except Exception as exc:  # noqa: BLE001
                    self._fail(ck, exc)

        if "zerodce" in s.enabled_backends:
            ck = s.zerodce_checkpoint
            try:
                _, path = weight_path(ck)
                self.enhancers["zerodce"] = {
                    ck: ZeroDCEEnhancer(build_zerodce(path, self.device), ck, self.device, max_side=s.zerodce_max_side)
                }
                log.info("loaded zerodce checkpoint", extra={"checkpoint": ck})
            except Exception as exc:  # noqa: BLE001
                self._fail(ck, exc)

        if "classical" in s.enabled_backends:
            self.enhancers["classical"] = {
                None: ClassicalEnhancer(
                    clip_limit=s.clahe_clip_limit, grid=s.clahe_grid, target=s.classical_target, max_gain=s.max_gain
                )
            }

        self.denoisers["opencv"] = OpenCVDenoiser()
        if s.enable_scunet and s.denoise_backend in ("auto", "scunet"):
            ck = s.scunet_checkpoint
            try:
                _, path = weight_path(ck)
                self.denoisers["scunet"] = SCUNetDenoiser(
                    build_scunet(path, self.device), self.device, s.denoise_tile_size, s.denoise_tile_overlap
                )
                log.info("loaded scunet checkpoint", extra={"checkpoint": ck})
            except Exception as exc:  # noqa: BLE001
                self._fail(ck, exc)

        self.iqa = NoReferenceIQA(self.device, s.iqa_max_side)
        if s.iqa_enabled:
            self.iqa.warmup()

        missing = [r for r in s.required_backends if not self._is_available(r)]
        if missing:
            raise RuntimeError(
                f"required backends failed to load: {missing}; failures: {self.failures}"
            )

    def _fail(self, name: str, exc: Exception) -> None:
        self.failures[name] = f"{type(exc).__name__}: {exc}"
        log.error("failed to load model", extra={"model": name, "error": self.failures[name]})

    def _is_available(self, name: str) -> bool:
        if name in self.enhancers or name in self.denoisers:
            return True
        return any(name in cks for cks in self.enhancers.values())

    def enhancer(self, method: str, checkpoint: str | None) -> Enhancer:
        cks = self.enhancers.get(method)
        if not cks:
            reason = {k: v for k, v in self.failures.items()} or "backend not enabled"
            raise BackendUnavailableError(
                f"method '{method}' is not loaded", details={"method": method, "reason": reason}
            )
        if method == "retinexformer":
            ck = checkpoint or self.settings.retinexformer_default_checkpoint
            if ck not in cks:
                if checkpoint is None:
                    ck = next(iter(cks))
                else:
                    raise BackendUnavailableError(
                        f"checkpoint '{checkpoint}' is not loaded",
                        details={"loaded": list(cks), "failure": self.failures.get(checkpoint)},
                    )
            return cks[ck]
        if checkpoint is not None and checkpoint not in cks:
            raise BackendUnavailableError(
                f"checkpoint '{checkpoint}' is not available for method '{method}'",
                details={"loaded": [c for c in cks if c]},
            )
        return next(iter(cks.values()))

    def denoiser(self) -> Denoiser:
        pref = self.settings.denoise_backend
        if pref in ("auto", "scunet") and "scunet" in self.denoisers:
            return self.denoisers["scunet"]
        return self.denoisers["opencv"]

    def describe(self) -> dict:
        s = self.settings
        return {
            "device": str(self.device),
            "default_method": s.default_method,
            "enhancers": {
                m: {
                    "checkpoints": [c for c in cks if c is not None],
                    "default_checkpoint": (
                        s.retinexformer_default_checkpoint if m == "retinexformer" else next(iter(cks), None)
                    ),
                }
                for m, cks in self.enhancers.items()
            },
            "auto_candidates": [c.label for c, _ in self.auto_candidates(None, None)],
            "denoisers": list(self.denoisers),
            "active_denoiser": self.denoiser().name,
            "iqa": {"enabled": s.iqa_enabled, "error": self.iqa.error if self.iqa else None},
            "failures": self.failures,
        }

    def pipeline_config(self) -> PipelineConfig:
        s = self.settings
        return PipelineConfig(
            knee=s.knee,
            clip_tolerance=s.clip_tolerance,
            max_gain=s.max_gain,
            mask_radius=s.highlight_mask_radius,
            mask_eps=s.highlight_mask_eps,
            ae_target_low=s.ae_target_low,
            ae_target_high=s.ae_target_high,
            ae_max_gain=s.ae_max_gain,
            ae_allow_darken=s.ae_allow_darken,
            sharpen_amount=s.sharpen_amount,
            sharpen_sigma=s.sharpen_sigma,
            dark_threshold=s.dark_threshold,
            noise_threshold=s.noise_threshold,
            pre_gain_targets=dict(s.pre_gain_targets),
            pre_gain_max=s.pre_gain_max,
            ae_white_target=s.ae_white_target,
            ae_bg_min_fraction=s.ae_bg_min_fraction,
            ae_white_max_input=s.ae_white_max_input,
        )

    def auto_candidates(self, checkpoint: str | None, preserve_color: bool | None) -> list[tuple[Candidate, Enhancer]]:
        out: list[tuple[Candidate, Enhancer]] = []
        for spec in self.settings.auto_candidates:
            cand = Candidate.parse(spec)
            if checkpoint is not None and cand.checkpoint != checkpoint:
                continue
            if preserve_color is not None and cand.preserve_color != preserve_color:
                continue
            cks = self.enhancers.get(cand.method, {})
            key = cand.checkpoint if cand.method != "classical" else None
            if key in cks:
                out.append((cand, cks[key]))
        return out

    def resolve_params(self, rgb: np.ndarray, params: EnhanceParams) -> tuple[EnhanceParams, Enhancer, dict | None]:
        """Turn 'auto' choices into a concrete method/checkpoint/color mode/strength for this image."""
        s = self.settings
        if params.method != "auto":
            pc = s.preserve_color if params.preserve_color is None else params.preserve_color
            enhancer = self.enhancer(params.method, params.checkpoint)
            params = replace(params, preserve_color=pc)
            if params.strength is not None:
                return params, enhancer, None
            candidates = [(Candidate(params.method, enhancer.checkpoint, pc), enhancer)]
        else:
            candidates = self.auto_candidates(params.checkpoint, params.preserve_color)
        if not candidates:
            raise BackendUnavailableError(
                "no auto candidate is loaded",
                details={"candidates": s.auto_candidates, "failures": self.failures},
            )
        weights = DEFAULT_WEIGHTS | dict(s.auto_weights)
        if not s.auto_use_niqe:
            weights["niqe"] = 0.0
        best, strength, info = select_candidate(
            rgb,
            candidates,
            params,
            self.pipeline_config(),
            self.denoiser(),
            self.iqa if s.iqa_enabled and s.auto_use_niqe else None,
            weights=weights,
            preview_side=s.auto_preview_side,
            strengths=list(s.auto_strengths),
        )
        enhancer = dict((c, e) for c, e in candidates)[best]
        resolved = replace(
            params,
            method=best.method,
            checkpoint=best.checkpoint,
            preserve_color=best.preserve_color,
            strength=params.strength if strength is None else strength,
        )
        return resolved, enhancer, info


def available_memory_bytes() -> int | None:
    """Free RAM for this process: min(MemAvailable, cgroup v2 limit - usage); None if unknown."""
    vals: list[int] = []
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    vals.append(int(line.split()[1]) * 1024)
                    break
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        if limit != "max":
            with open("/sys/fs/cgroup/memory.current") as f:
                vals.append(int(limit) - int(f.read().strip()))
    except (OSError, ValueError):
        pass
    return max(0, min(vals)) if vals else None


def memory_fit_scale(pixels: int, bytes_per_pixel: int, headroom: float, available: int | None) -> float:
    """Linear scale factor (<= 1) so the working set of a `pixels`-sized image fits in free RAM."""
    if available is None:
        return 1.0
    need = pixels * bytes_per_pixel
    budget = available * headroom
    return 1.0 if need <= budget else float(np.sqrt(budget / need))


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


@dataclass
class ProcessedImage:
    data: bytes
    media_type: str
    width: int
    height: int
    metrics: dict
    pipeline: dict
    timings_ms: dict


def process_image(registry: ModelRegistry, data: bytes, params: EnhanceParams) -> ProcessedImage:
    """Synchronous end-to-end job: decode -> pipeline -> encode -> metrics."""
    s = registry.settings
    t0 = time.perf_counter()
    if params.method != "auto":
        registry.enhancer(params.method, params.checkpoint)  # fail fast before decoding
    decoded = decode_image(data, s.max_input_pixels)
    t_decode = (time.perf_counter() - t0) * 1000.0

    h0, w0 = decoded.rgb.shape[:2]
    scale = memory_fit_scale(h0 * w0, s.memory_bytes_per_pixel, s.memory_headroom, available_memory_bytes())
    if scale < 1.0:
        if h0 * w0 * scale * scale < 0.25e6:
            raise BackendUnavailableError("Not enough free memory to process this image right now; retry later.")
        decoded.rgb = _resize(decoded.rgb, scale)
        if decoded.alpha is not None:
            decoded.alpha = _resize(decoded.alpha, scale)
        log.warning("downscaled %dx%d by %.2f to fit free memory", w0, h0, scale)
    downscaled = {"from": [w0, h0], "to": [decoded.rgb.shape[1], decoded.rgb.shape[0]]} if scale < 1.0 else None

    t_auto0 = time.perf_counter()
    params, enhancer, auto_info = registry.resolve_params(decoded.rgb, params)
    t_auto = (time.perf_counter() - t_auto0) * 1000.0

    result = run_pipeline(decoded.rgb, params, registry.pipeline_config(), enhancer, registry.denoiser())
    result.info["resolved"] = {
        "method": params.method,
        "checkpoint": params.checkpoint,
        "preserve_color": params.preserve_color,
        "strength": params.strength,
    }
    if auto_info is not None:
        result.info["auto"] = auto_info
    result.info["downscaled_for_memory"] = downscaled

    t1 = time.perf_counter()
    payload, media_type = encode_image(
        result.image,
        params.output_format,
        quality=params.quality,
        alpha=decoded.alpha,
        exif=None if params.strip_exif else decoded.exif,
    )
    t_encode = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    before, after = luminance_stats(decoded.rgb), luminance_stats(result.image)
    if params.iqa and registry.iqa is not None:
        before |= registry.iqa.score(decoded.rgb)
        after |= registry.iqa.score(result.image)
    t_metrics = (time.perf_counter() - t2) * 1000.0

    timings = {"decode": round(t_decode, 1), "auto_select": round(t_auto, 1), **result.timings_ms, "encode": round(t_encode, 1), "metrics": round(t_metrics, 1)}
    timings["total"] = round((time.perf_counter() - t0) * 1000.0, 1)
    h, w = result.image.shape[:2]
    return ProcessedImage(
        data=payload,
        media_type=media_type,
        width=w,
        height=h,
        metrics={"before": before, "after": after},
        pipeline=result.info | {"icc_converted": decoded.icc_converted, "source_format": decoded.source_format},
        timings_ms=timings,
    )


class InferenceRunner:
    """Bounded-concurrency executor with a per-request deadline.

    A timed-out job cannot be interrupted inside PyTorch/OpenCV; its concurrency slot is
    released only when the work actually finishes, so timeouts never cause overload.
    """

    def __init__(self, max_concurrency: int, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self._sem = asyncio.Semaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="infer")

    async def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_s
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self.timeout_s)
        except asyncio.TimeoutError as exc:
            raise RequestTimeoutError(
                f"server busy: no inference slot within {self.timeout_s:.0f}s"
            ) from exc
        try:
            ctx = contextvars.copy_context()
            fut = loop.run_in_executor(self._executor, ctx.run, fn, *args)
        except BaseException:
            self._sem.release()
            raise
        fut.add_done_callback(lambda _f: self._sem.release())
        remaining = max(0.001, deadline - loop.time())
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise RequestTimeoutError(f"processing exceeded {self.timeout_s:.0f}s") from exc

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)
