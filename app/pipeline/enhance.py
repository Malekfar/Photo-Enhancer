"""Low-light enhancement backends: retinexformer, zerodce, classical."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.pipeline.color import fast_median, luminance, scale_to_max_pixels
from app.pipeline.tiling import guided_upsample, run_padded, tiled_apply, to_numpy, to_tensor


@dataclass
class EnhanceResult:
    image: np.ndarray  # HxWx3 float32, >= 0, may exceed 1 (rolled off later)
    info: dict = field(default_factory=dict)


class Enhancer(ABC):
    method: str
    checkpoint: str | None = None

    @abstractmethod
    def enhance(self, rgb: np.ndarray) -> EnhanceResult: ...

    def describe(self) -> dict:
        return {"method": self.method, "checkpoint": self.checkpoint}


class RetinexformerEnhancer(Enhancer):
    method = "retinexformer"

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint: str,
        device: torch.device,
        *,
        max_model_pixels: int,
        large_mode: str,
        tile: int,
        overlap: int,
        guided_radius: int,
        guided_eps: float,
    ) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.max_model_pixels = max_model_pixels
        self.large_mode = large_mode
        self.tile = tile
        self.overlap = overlap
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        self.multiple = getattr(model, "size_multiple", 4)

    def _forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.model(t.to(self.device))

    def _run(self, rgb: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            out = run_padded(self._forward, to_tensor(rgb, self.device), self.multiple)
        return np.maximum(to_numpy(out), 0.0)

    def enhance(self, rgb: np.ndarray) -> EnhanceResult:
        h, w = rgb.shape[:2]
        if h * w <= self.max_model_pixels:
            return EnhanceResult(self._run(rgb), {"mode": "full", "model_size": [w, h]})
        if self.large_mode == "tile":
            with torch.inference_mode():
                out = tiled_apply(
                    self._forward, to_tensor(rgb, torch.device("cpu")), self.tile, self.overlap, self.multiple
                )
            return EnhanceResult(np.maximum(to_numpy(out), 0.0), {"mode": "tile", "tile": self.tile})
        lh, lw = scale_to_max_pixels(h, w, self.max_model_pixels)
        low = cv2.resize(rgb, (lw, lh), interpolation=cv2.INTER_AREA)
        out_low = self._run(low)
        full = guided_upsample(rgb, low, out_low, self.guided_radius, self.guided_eps)
        return EnhanceResult(np.maximum(full, 0.0), {"mode": "guided", "model_size": [lw, lh]})


class ZeroDCEEnhancer(Enhancer):
    method = "zerodce"

    def __init__(self, model: torch.nn.Module, checkpoint: str, device: torch.device, *, max_side: int) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.max_side = max_side

    def enhance(self, rgb: np.ndarray) -> EnhanceResult:
        h, w = rgb.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        lh, lw = max(1, round(h * scale)), max(1, round(w * scale))
        low = rgb if scale == 1.0 else cv2.resize(rgb, (lw, lh), interpolation=cv2.INTER_AREA)
        out = np.empty_like(rgb)
        with torch.inference_mode():
            x_r = self.model.curve_params(to_tensor(low, self.device))
            # Curve maps are low-frequency; apply the per-pixel curves at full resolution
            # in row chunks to bound memory on very large images.
            chunk = max(1, int(4_000_000 // max(w, 1)))
            for y0 in range(0, h, chunk):
                y1 = min(h, y0 + chunk)
                x_full = to_tensor(rgb[y0:y1], self.device)
                if scale == 1.0:
                    r = x_r[..., y0:y1, :]
                else:
                    r = self._upsample_rows(x_r, h, w, y0, y1)
                res = self.model.apply_curves(x_full, r, self.model.curve_iterations)
                out[y0:y1] = to_numpy(res)
        return EnhanceResult(np.clip(out, 0.0, 1.0), {"mode": "curves", "model_size": [lw, lh]})

    @staticmethod
    def _upsample_rows(x_r: torch.Tensor, h: int, w: int, y0: int, y1: int) -> torch.Tensor:
        # Bilinear (align_corners=False) upsampling evaluated only for rows [y0, y1),
        # identical to slicing a full-size F.interpolate result.
        lh = x_r.shape[-2]
        ys = (torch.arange(y0, y1, device=x_r.device, dtype=torch.float32) + 0.5) * (lh / h) - 0.5
        ys = ys.clamp(0, lh - 1)
        i0 = ys.floor().long()
        i1 = (i0 + 1).clamp(max=lh - 1)
        fy = (ys - i0.float()).view(1, 1, -1, 1)
        rows = x_r[..., i0, :] * (1 - fy) + x_r[..., i1, :] * fy
        # Height is unchanged, so this only interpolates horizontally.
        return F.interpolate(rows, size=(y1 - y0, w), mode="bilinear", align_corners=False)


class ClassicalEnhancer(Enhancer):
    """Adaptive gamma on luminance (color-preserving) + CLAHE on L* in CIELAB."""

    method = "classical"

    def __init__(self, *, clip_limit: float, grid: int, target: float, max_gain: float) -> None:
        self.clip_limit = clip_limit
        self.grid = grid
        self.target = target
        self.max_gain = max_gain

    def enhance(self, rgb: np.ndarray) -> EnhanceResult:
        y = luminance(rgb)
        med = float(np.clip(fast_median(y), 1e-3, 0.99))
        gamma = float(np.clip(np.log(self.target) / np.log(med), 0.25, 1.0))
        y_new = np.power(np.maximum(y, 0.0), gamma)
        gain = np.minimum(y_new / np.maximum(y, 1e-4), self.max_gain)
        lifted = rgb * gain[..., None]
        # Bring out-of-range pixels back into gamut by scaling (keeps hue and saturation).
        peak = np.maximum(lifted.max(axis=-1, keepdims=True), 1.0)
        lifted = (lifted / peak).astype(np.float32)

        lab = cv2.cvtColor(lifted, cv2.COLOR_RGB2LAB)
        L16 = np.clip(lab[..., 0] / 100.0 * 65535.0, 0, 65535).astype(np.uint16)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.grid, self.grid))
        lab[..., 0] = clahe.apply(L16).astype(np.float32) / 65535.0 * 100.0
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return EnhanceResult(np.clip(out, 0.0, 1.0), {"mode": "classical", "gamma": round(gamma, 4)})
