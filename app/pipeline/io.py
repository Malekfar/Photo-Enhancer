"""Image decoding/encoding: EXIF orientation, ICC -> sRGB, float32 [0,1], format output."""
from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageCms, ImageFile, ImageOps, UnidentifiedImageError

from app.errors import InvalidImageError, PayloadTooLargeError, UnsupportedMediaTypeError

log = logging.getLogger(__name__)

SUPPORTED_INPUT = {"JPEG": "jpeg", "PNG": "png", "WEBP": "webp"}
MEDIA_TYPES = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
EXTENSIONS = {"jpeg": "jpg", "png": "png", "webp": "webp"}

_SRGB_PROFILE = ImageCms.createProfile("sRGB")
SRGB_ICC = ImageCms.ImageCmsProfile(_SRGB_PROFILE).tobytes()
_ORIENTATION_TAG = 0x0112

# The pixel limit is enforced explicitly from the header (before any pixel data is decoded)
# with a configurable value, replacing Pillow's fixed global guard.
Image.MAX_IMAGE_PIXELS = None


@dataclass
class DecodedImage:
    rgb: np.ndarray  # HxWx3 float32 sRGB in [0, 1]
    alpha: np.ndarray | None  # HxW float32 in [0, 1]
    source_format: str
    exif: bytes | None  # orientation tag removed (pixels are already upright)
    icc_converted: bool

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


def _to_srgb(img: Image.Image, icc: bytes | None) -> tuple[Image.Image, bool]:
    if not icc:
        return img.convert("RGB"), False
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        out = ImageCms.profileToProfile(
            img,
            src,
            _SRGB_PROFILE,
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode="RGB",
        )
        return out, True
    except (ImageCms.PyCMSError, OSError, ValueError) as exc:
        log.warning("embedded ICC profile could not be applied; assuming sRGB", extra={"error": str(exc)})
        return img.convert("RGB"), False


def decode_image(data: bytes, max_pixels: int) -> DecodedImage:
    if not data:
        raise InvalidImageError("empty file")
    try:
        img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as exc:
        raise UnsupportedMediaTypeError("file is not a recognised image") from exc
    except Exception as exc:  # noqa: BLE001 - Pillow raises many types for corrupt headers
        raise InvalidImageError(f"cannot read image header: {exc}") from exc

    fmt = img.format
    if fmt not in SUPPORTED_INPUT:
        raise UnsupportedMediaTypeError(
            f"unsupported image format {fmt!r}; accepted: JPEG, PNG, WebP",
            details={"format": fmt},
        )
    w, h = img.size
    if w <= 0 or h <= 0:
        raise InvalidImageError("image has zero size")
    if w * h > max_pixels:
        raise PayloadTooLargeError(
            f"image is {w}x{h} = {w * h / 1e6:.1f} MP; limit is {max_pixels / 1e6:.1f} MP",
            details={"width": w, "height": h, "max_pixels": max_pixels},
        )
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)

    try:
        # Third-party libraries sometimes flip this global; truncated uploads must fail.
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img.load()
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError(f"corrupt or truncated image data: {exc}") from exc

    icc = img.info.get("icc_profile")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception as exc:  # noqa: BLE001 - malformed EXIF must not kill the request
        log.warning("EXIF orientation could not be applied", extra={"error": str(exc)})

    exif_bytes: bytes | None = None
    try:
        exif = img.getexif()
        if len(exif):
            exif.pop(_ORIENTATION_TAG, None)
            exif_bytes = exif.tobytes()
    except Exception:  # noqa: BLE001
        exif_bytes = None

    alpha: np.ndarray | None = None
    mode = img.mode
    if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
        if mode in ("LA", "PA", "P"):
            base, converted = rgba.convert("RGB"), False
        else:
            base, converted = _to_srgb(rgba.convert("RGB"), icc)
        rgb = np.asarray(base, dtype=np.float32) / 255.0
    elif mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img).astype(np.float32)
        scale = 65535.0 if mode.startswith("I;16") or (mode == "I" and arr.max() > 255) else 255.0
        if mode == "F":
            scale = max(1.0, float(arr.max()))
        g = np.clip(arr / scale, 0.0, 1.0)
        rgb, converted = np.repeat(g[..., None], 3, axis=2), False
    else:
        if mode in ("RGB", "CMYK", "L"):
            base, converted = _to_srgb(img, icc)
        else:
            base, converted = img.convert("RGB"), False
        rgb = np.asarray(base, dtype=np.float32) / 255.0

    return DecodedImage(
        rgb=np.ascontiguousarray(rgb, dtype=np.float32),
        alpha=alpha,
        source_format=SUPPORTED_INPUT[fmt],
        exif=exif_bytes,
        icc_converted=converted,
    )


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def encode_image(
    rgb: np.ndarray,
    fmt: str,
    *,
    quality: int,
    alpha: np.ndarray | None = None,
    exif: bytes | None = None,
) -> tuple[bytes, str]:
    """Encode float RGB [0,1] to bytes. Always embeds an sRGB ICC profile."""
    fmt = fmt.lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in MEDIA_TYPES:
        raise ValueError(f"unsupported output format {fmt}")
    if alpha is not None and fmt == "jpeg":
        a = alpha[..., None]
        rgb = rgb * a + (1.0 - a)  # JPEG has no alpha: flatten onto white
        alpha = None
    arr = to_uint8(rgb)
    if alpha is not None:
        img = Image.fromarray(np.ascontiguousarray(np.dstack([arr, to_uint8(alpha)])))
    else:
        img = Image.fromarray(np.ascontiguousarray(arr))

    kwargs: dict = {"icc_profile": SRGB_ICC}
    if exif:
        kwargs["exif"] = exif
    if fmt == "jpeg":
        kwargs.update(quality=quality, subsampling=0, optimize=True)
        pil_fmt = "JPEG"
    elif fmt == "png":
        kwargs.update(compress_level=6)
        pil_fmt = "PNG"
    else:
        kwargs.update(quality=quality, method=6, lossless=quality >= 100)
        pil_fmt = "WEBP"
    buf = io.BytesIO()
    img.save(buf, format=pil_fmt, **kwargs)
    return buf.getvalue(), MEDIA_TYPES[fmt]
