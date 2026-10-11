"""Padding, cropping, tiled inference with feathered blending, and guided upsampling."""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad NCHW tensor on bottom/right to a multiple of `multiple`. Returns (padded, (h, w))."""
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x, (h, w)
    mode = "reflect" if ph < h and pw < w else "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=mode), (h, w)


def crop_to(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    return x[..., : hw[0], : hw[1]]


def run_padded(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, multiple: int) -> torch.Tensor:
    padded, hw = pad_to_multiple(x, multiple)
    return crop_to(fn(padded), hw)


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def _ramp(length: int, overlap: int) -> np.ndarray:
    """1D feather weight: linear ramps of `overlap` px at both ends, never zero."""
    w = np.ones(length, dtype=np.float32)
    n = min(overlap, length // 2)
    if n > 0:
        r = (np.arange(n, dtype=np.float32) + 0.5) / n
        w[:n] = r
        w[length - n :] = r[::-1]
    return w


def tiled_apply(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    tile: int,
    overlap: int,
    multiple: int = 1,
) -> torch.Tensor:
    """Apply `fn` (NCHW -> NCHW, same spatial size) tile by tile with feathered blending.

    Each tile is padded to `multiple` and cropped back. Accumulation happens on CPU in float32
    so device memory only ever holds one tile.
    """
    n, _, h, w = x.shape
    if h <= tile and w <= tile:
        return run_padded(fn, x, multiple).float().cpu()
    stride = tile - overlap
    ys, xs = _starts(h, tile, stride), _starts(w, tile, stride)
    out: torch.Tensor | None = None
    weight = torch.zeros((1, 1, h, w), dtype=torch.float32)
    for y0 in ys:
        th = min(tile, h - y0)
        wy = _ramp(th, overlap)
        for x0 in xs:
            tw = min(tile, w - x0)
            wx = _ramp(tw, overlap)
            patch = x[..., y0 : y0 + th, x0 : x0 + tw]
            res = run_padded(fn, patch, multiple).float().cpu()
            if out is None:
                out = torch.zeros((n, res.shape[1], h, w), dtype=torch.float32)
            wmap = torch.from_numpy(np.outer(wy, wx))[None, None]
            out[..., y0 : y0 + th, x0 : x0 + tw] += res * wmap
            weight[..., y0 : y0 + th, x0 : x0 + tw] += wmap
    assert out is not None
    return out / weight


def guided_upsample(
    src_full: np.ndarray,
    src_low: np.ndarray,
    out_low: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Transfer a low-res enhancement to full resolution without resampling the content.

    Per channel, fit a local affine model out_low ≈ a * src_low + b (guided-filter
    coefficients), bilinearly upsample (a, b) and apply them to the full-resolution input.
    All fine structure (text, edges, texture) therefore comes from the original pixels;
    the model only contributes a smooth, edge-aware tone/color mapping.

    The slope `a` is regularised towards the local gain mean(out)/mean(src) rather than
    towards 0, so low-variance (dark, flat) regions keep their texture scaled by the gain
    instead of being flattened to their mean.
    """
    H, W = src_full.shape[:2]
    out = np.empty((H, W, 3), dtype=np.float32)
    k = (2 * radius + 1, 2 * radius + 1)

    def box(z: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(z, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)

    for c in range(3):
        I = src_low[..., c].astype(np.float32)
        p = out_low[..., c].astype(np.float32)
        mI, mp = box(I), box(p)
        cov = box(I * p) - mI * mp
        var = box(I * I) - mI * mI
        gain = np.clip(mp / (mI + 1e-4), 0.0, 64.0)
        a = (cov + eps * gain) / (var + eps)
        b = mp - a * mI
        a = cv2.resize(box(a), (W, H), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(box(b), (W, H), interpolation=cv2.INTER_LINEAR)
        out[..., c] = a * src_full[..., c] + b
    return out


def to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(device=device, dtype=torch.float32)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t[0].detach().float().cpu().numpy().transpose(1, 2, 0).astype(np.float32, copy=False)
