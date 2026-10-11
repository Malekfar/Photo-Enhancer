"""Output quality metrics: luminance statistics, clipping, no-reference IQA (pyiqa)."""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np
import torch

from app.pipeline.color import CLIP_LEVEL, luminance, subsample, value_channel

log = logging.getLogger(__name__)


def luminance_stats(rgb: np.ndarray) -> dict:
    y = luminance(rgb)
    return {
        "mean_luminance": round(float(y.mean()), 5),
        "median_luminance": round(float(np.median(subsample(y, 4_000_000))), 5),
        "clipped_pct": round(float(np.mean(value_channel(rgb) > CLIP_LEVEL)) * 100.0, 4),
    }


class NoReferenceIQA:
    """NIQE and BRISQUE via pyiqa (lower is better for both).

    Computed on a copy downscaled to `max_side` (both metrics are defined on natural-scene
    statistics at moderate resolution, and full 24 MP evaluation would dominate latency).
    pyiqa is loaded lazily; if it is unavailable the scores are null with an error string.
    """

    names = ("niqe", "brisque")

    def __init__(self, device: torch.device, max_side: int) -> None:
        self.device = device
        self.max_side = max_side
        self._metrics: dict | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        if self._metrics is not None or self._error is not None:
            return
        try:
            import pyiqa  # noqa: PLC0415 - heavy optional import

            self._metrics = {n: pyiqa.create_metric(n, device=self.device, as_loss=False) for n in self.names}
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            log.warning("no-reference IQA unavailable", extra={"error": self._error})

    def warmup(self) -> bool:
        with self._lock:
            self._ensure()
        return self._metrics is not None

    @property
    def error(self) -> str | None:
        return self._error

    def score(self, rgb: np.ndarray) -> dict:
        h, w = rgb.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        img = rgb if s == 1.0 else cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        img = np.clip(img, 0.0, 1.0)
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float().to(self.device)
        with self._lock:
            self._ensure()
            if self._metrics is None:
                return {n: None for n in self.names} | {"iqa_error": self._error}
            out: dict = {}
            with torch.inference_mode():
                for n, metric in self._metrics.items():
                    try:
                        out[n] = round(float(metric(t).item()), 4)
                    except Exception as exc:  # noqa: BLE001
                        out[n] = None
                        out[f"{n}_error"] = f"{type(exc).__name__}: {exc}"
            return out
