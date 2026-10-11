from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from app.pipeline.tiling import crop_to, guided_upsample, pad_to_multiple, run_padded, tiled_apply


@pytest.mark.parametrize("h,w,m", [(37, 53, 4), (64, 64, 64), (65, 1, 8), (100, 129, 64)])
def test_pad_then_crop_is_exact(h, w, m):
    x = torch.rand(1, 3, h, w)
    p, hw = pad_to_multiple(x, m)
    assert p.shape[-2] % m == 0 and p.shape[-1] % m == 0
    assert p.shape[-2] - h < m and p.shape[-1] - w < m
    assert hw == (h, w)
    assert torch.equal(crop_to(p, hw), x)


def test_pad_noop_when_aligned():
    x = torch.rand(1, 3, 32, 64)
    p, _ = pad_to_multiple(x, 32)
    assert p is x


def test_pad_reflect_values():
    x = torch.arange(6, dtype=torch.float32).view(1, 1, 2, 3)
    p, _ = pad_to_multiple(x, 4)
    assert p.shape == (1, 1, 4, 4)
    # Bottom rows come from reflection/replication of the existing ones.
    assert set(p[0, 0, 2:, :3].flatten().tolist()) <= set(x.flatten().tolist())


def test_run_padded_feeds_aligned_input():
    seen = {}

    def fn(t):
        seen["shape"] = tuple(t.shape[-2:])
        assert t.shape[-2] % 8 == 0 and t.shape[-1] % 8 == 0
        return t * 2

    x = torch.rand(1, 3, 13, 21)
    out = run_padded(fn, x, 8)
    assert seen["shape"] == (16, 24)
    assert torch.allclose(out, x * 2)


def test_tiled_identity_is_exact():
    x = torch.rand(1, 3, 300, 451)
    out = tiled_apply(lambda t: t, x, tile=128, overlap=32, multiple=8)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6)


def _blur(t: torch.Tensor) -> torch.Tensor:
    k = torch.ones(3, 1, 5, 5) / 25.0
    return F.conv2d(F.pad(t, (2, 2, 2, 2), mode="reflect"), k, groups=3)


def test_tiling_has_no_seams():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 320, 480)
    full = _blur(x)
    tiled = tiled_apply(_blur, x, tile=128, overlap=48, multiple=4)
    diff = (full - tiled).abs()
    # Pixels near a tile edge see wrong (reflected) context, but feathering gives them a
    # weight of ~1% there, so the residual stays far below one 8-bit code value (3.9e-3).
    assert diff.mean() < 2e-4
    assert diff.max() < 1e-2
    stride = 128 - 48
    cols = list(range(stride, 480 - 128, stride))
    assert diff[..., cols].mean() < 2e-3


def test_tiling_small_input_single_pass():
    x = torch.rand(1, 3, 50, 60)
    calls = []

    def fn(t):
        calls.append(t.shape)
        return t

    tiled_apply(fn, x, tile=128, overlap=16, multiple=4)
    assert len(calls) == 1


def test_guided_upsample_reproduces_global_gain():
    rng = np.random.default_rng(0)
    full = rng.random((200, 300, 3)).astype(np.float32) * 0.3
    import cv2

    low = cv2.resize(full, (75, 50), interpolation=cv2.INTER_AREA)
    out = guided_upsample(full, low, low * 2.0, radius=2, eps=1e-4)
    assert out.shape == full.shape
    np.testing.assert_allclose(out, full * 2.0, atol=1e-3)


def test_guided_upsample_keeps_full_res_detail():
    import cv2

    rng = np.random.default_rng(1)
    full = (0.1 + 0.05 * rng.random((256, 256, 3))).astype(np.float32)
    full[100:110, :, :] = 0.02  # thin dark line (1/4 of a low-res pixel tall after 4x)
    low = cv2.resize(full, (64, 64), interpolation=cv2.INTER_AREA)
    out = guided_upsample(full, low, np.clip(low * 3.0, 0, 1), radius=2, eps=1e-4)
    # Detail present at full resolution must survive (not be averaged away).
    assert out[105, :, 0].mean() < out[90, :, 0].mean() * 0.5
