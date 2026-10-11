from __future__ import annotations

import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from app.errors import InvalidImageError, PayloadTooLargeError, UnsupportedMediaTypeError
from app.pipeline.io import SRGB_ICC, decode_image, encode_image

MAX = 24_000_000


def _save(arr: np.ndarray, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, **kw)
    return buf.getvalue()


def _gradient(h: int = 32, w: int = 48) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.dstack([x * 255 // (w - 1), y * 255 // (h - 1), np.full_like(x, 128)]).astype(np.uint8)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_decode_supported_formats(fmt):
    arr = _gradient()
    d = decode_image(_save(arr, fmt, **({"quality": 95} if fmt != "PNG" else {})), MAX)
    assert d.rgb.dtype == np.float32
    assert d.rgb.shape == (32, 48, 3)
    assert 0.0 <= d.rgb.min() and d.rgb.max() <= 1.0
    assert d.source_format == fmt.lower()
    if fmt == "PNG":
        np.testing.assert_allclose(d.rgb, arr / 255.0, atol=1e-6)


def test_exif_orientation_applied():
    arr = np.zeros((20, 40, 3), dtype=np.uint8)
    arr[0, 0] = (255, 0, 0)  # stored top-left
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW for display
    d = decode_image(_save(arr, "PNG", exif=exif.tobytes()), MAX)
    assert d.rgb.shape == (40, 20, 3)
    np.testing.assert_allclose(d.rgb[0, -1], [1.0, 0.0, 0.0], atol=1e-6)  # now top-right
    if d.exif:
        remaining = Image.Exif()
        remaining.load(d.exif)
        assert 0x0112 not in remaining


def test_embedded_srgb_icc_is_applied_without_shift():
    arr = _gradient()
    d = decode_image(_save(arr, "PNG", icc_profile=SRGB_ICC), MAX)
    assert d.icc_converted
    assert np.abs(d.rgb - arr / 255.0).max() <= 2.0 / 255.0


def test_alpha_preserved_and_jpeg_flattens_on_white():
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., :3] = 30
    rgba[:8, :, 3] = 255  # top half opaque, bottom half transparent
    d = decode_image(_save(rgba, "PNG"), MAX)
    assert d.alpha is not None and d.alpha.shape == (16, 16)
    png, mt = encode_image(d.rgb, "png", quality=95, alpha=d.alpha)
    assert mt == "image/png"
    assert Image.open(io.BytesIO(png)).mode == "RGBA"
    jpg, _ = encode_image(d.rgb, "jpeg", quality=95, alpha=d.alpha)
    back = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")).astype(int)
    assert back[12, 8].min() >= 250  # transparent -> white
    assert back[2, 8].max() <= 40


def test_pixel_limit_rejected():
    data = _save(np.zeros((1100, 1000, 3), dtype=np.uint8), "PNG")
    with pytest.raises(PayloadTooLargeError):
        decode_image(data, 1_000_000)


def _png_with_declared_size(width: int, height: int) -> bytes:
    data = bytearray(_save(np.zeros((1, 1, 3), dtype=np.uint8), "PNG"))
    # IHDR chunk: 8-byte signature, 4-byte length, b"IHDR", 13 bytes of data, 4-byte CRC
    ihdr_start = 8 + 4
    payload = bytes(data[ihdr_start : ihdr_start + 4 + 13])
    payload = payload[:4] + struct.pack(">II", width, height) + payload[12:]
    crc = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    data[ihdr_start : ihdr_start + 4 + 13 + 4] = payload + crc
    return bytes(data)


def test_decompression_bomb_rejected_from_header():
    bomb = _png_with_declared_size(100_000, 100_000)
    with pytest.raises(PayloadTooLargeError):
        decode_image(bomb, MAX)


def test_garbage_and_unsupported_formats():
    with pytest.raises(UnsupportedMediaTypeError):
        decode_image(b"definitely not an image", MAX)
    gif = _save(np.zeros((8, 8, 3), dtype=np.uint8), "GIF")
    with pytest.raises(UnsupportedMediaTypeError):
        decode_image(gif, MAX)
    with pytest.raises(InvalidImageError):
        decode_image(b"", MAX)


def test_truncated_image_rejected():
    data = _save(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8), "JPEG", quality=95)
    with pytest.raises(InvalidImageError):
        decode_image(data[: len(data) // 2], MAX)


@pytest.mark.parametrize("fmt,mt", [("jpeg", "image/jpeg"), ("png", "image/png"), ("webp", "image/webp")])
def test_encode_embeds_srgb_and_roundtrips(fmt, mt):
    rgb = _gradient().astype(np.float32) / 255.0
    data, media = encode_image(rgb, fmt, quality=95)
    assert media == mt
    img = Image.open(io.BytesIO(data))
    assert img.info.get("icc_profile") == SRGB_ICC
    assert img.size == (48, 32)
    assert not img.getexif()


def test_encode_exif_optional():
    exif = Image.Exif()
    exif[0x010F] = "TestMaker"
    rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
    data, _ = encode_image(rgb, "jpeg", quality=90, exif=exif.tobytes())
    assert Image.open(io.BytesIO(data)).getexif().get(0x010F) == "TestMaker"
