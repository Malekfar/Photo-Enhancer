"""Full-reference evaluation helpers and synthetic low-light data (used by tests/benchmark)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from app.pipeline.color import luminance

log = logging.getLogger(__name__)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.clip(a, 0, 1).astype(np.float64) - np.clip(b, 0, 1).astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM (Wang et al. 2004) on luminance, 11x11 Gaussian window, sigma 1.5."""
    x = luminance(np.clip(a, 0, 1)).astype(np.float64)
    y = luminance(np.clip(b, 0, 1)).astype(np.float64)
    c1, c2 = (0.01) ** 2, (0.03) ** 2

    def blur(z: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(z, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)

    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s[5:-5, 5:-5].mean()) if min(s.shape) > 10 else float(s.mean())


class LPIPS:
    """LPIPS (AlexNet) via pyiqa. Weights are fetched by pyiqa on first use."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self._metric = None
        self.error: str | None = None
        try:
            import pyiqa  # noqa: PLC0415

            self._metric = pyiqa.create_metric("lpips", device=self.device, as_loss=False)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("LPIPS unavailable", extra={"error": self.error})

    @property
    def available(self) -> bool:
        return self._metric is not None

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float | None:
        if self._metric is None:
            return None

        def t(z: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(np.clip(z, 0, 1).transpose(2, 0, 1)))[None].float().to(self.device)

        with torch.inference_mode():
            return float(self._metric(t(a), t(b)).item())


def darken(
    ref: np.ndarray,
    *,
    gamma: float,
    exposure: float,
    peak: float,
    read_noise: float,
    seed: int,
) -> np.ndarray:
    """Synthetic low-light: gamma (2.2-3.5) + exposure scale + Poisson-Gaussian noise.

    peak: photons at full scale (smaller = more shot noise). read_noise: Gaussian std in [0,1].
    """
    rng = np.random.default_rng(seed)
    clean = np.power(np.clip(ref, 0, 1), gamma) * exposure
    shot = rng.poisson(clean * peak).astype(np.float32) / peak
    noisy = shot + rng.normal(0.0, read_noise, size=clean.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class Degradation:
    gamma: float
    exposure: float
    peak: float
    read_noise: float

    @property
    def label(self) -> str:
        return f"g{self.gamma}_e{self.exposure}_p{int(self.peak)}_n{self.read_noise}"


DEGRADATIONS = (
    Degradation(gamma=2.2, exposure=0.5, peak=800.0, read_noise=0.004),
    Degradation(gamma=2.8, exposure=0.4, peak=400.0, read_noise=0.006),
    Degradation(gamma=3.5, exposure=0.3, peak=200.0, read_noise=0.008),
    Degradation(gamma=2.2, exposure=0.06, peak=100.0, read_noise=0.004),  # very dark
)


def make_product_scene(h: int = 512, w: int = 768, seed: int = 0) -> np.ndarray:
    """Deterministic, well-exposed e-commerce-like scene.

    White seamless background with clipped areas, a colored package with printed text, a
    logo, a fine fabric texture and a specular highlight: the structures the service must
    not alter.
    """
    rng = np.random.default_rng(seed)
    img = np.ones((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Background: pure white at the top fading to light gray (sweep).
    img *= (1.0 - 0.18 * (yy / h) ** 2)[..., None]
    # Soft shadow below the product.
    shadow = np.exp(-(((xx - w * 0.42) / (w * 0.22)) ** 2 + ((yy - h * 0.86) / (h * 0.05)) ** 2))
    img *= (1.0 - 0.35 * shadow)[..., None]
    # Package.
    x0, x1, y0, y1 = int(w * 0.18), int(w * 0.62), int(h * 0.18), int(h * 0.82)
    img[y0:y1, x0:x1] = np.array([0.80, 0.22, 0.18], dtype=np.float32)
    shade = np.linspace(1.0, 0.75, x1 - x0, dtype=np.float32)
    img[y0:y1, x0:x1] *= shade[None, :, None]
    # Printed label with text.
    lx0, lx1, ly0, ly1 = x0 + 20, x1 - 20, y0 + 30, y0 + 150
    img[ly0:ly1, lx0:lx1] = 0.95
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cv2.putText(u8, "ACME 250g", (lx0 + 12, ly0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(u8, "Organic Coffee - Lot 42", (lx0 + 12, ly0 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    # Logo.
    cv2.circle(u8, ((x0 + x1) // 2, y1 - 90), 45, (240, 200, 40), -1, cv2.LINE_AA)
    cv2.circle(u8, ((x0 + x1) // 2, y1 - 90), 45, (30, 30, 30), 3, cv2.LINE_AA)
    img = u8.astype(np.float32) / 255.0
    # Fabric swatch with fine texture.
    fx0, fx1, fy0, fy1 = int(w * 0.68), int(w * 0.92), int(h * 0.45), int(h * 0.80)
    weave = 0.5 + 0.5 * np.sin(xx[fy0:fy1, fx0:fx1] * 1.3) * np.sin(yy[fy0:fy1, fx0:fx1] * 1.3)
    base = np.array([0.15, 0.30, 0.55], dtype=np.float32)
    img[fy0:fy1, fx0:fx1] = base * (0.75 + 0.35 * weave[..., None])
    img[fy0:fy1, fx0:fx1] += rng.normal(0, 0.01, size=(fy1 - fy0, fx1 - fx0, 3)).astype(np.float32)
    # Specular highlight on the package (clipped).
    spec = np.exp(-(((xx - x0 - 40) / 18.0) ** 2 + ((yy - y1 + 60) / 60.0) ** 2))
    img = img + 0.9 * spec[..., None]
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def reference_images(extra_dir: Path | None = None, max_side: int = 768) -> dict[str, np.ndarray]:
    """Well-exposed references: synthetic product scenes, scikit-image photos (if
    installed), plus any images in `extra_dir`."""
    refs: dict[str, np.ndarray] = {
        "product_scene_a": make_product_scene(512, 768, seed=0),
        "product_scene_b": make_product_scene(600, 600, seed=1),
    }
    try:
        from skimage import data as skdata  # noqa: PLC0415

        for name in ("astronaut", "coffee", "chelsea"):
            try:
                refs[name] = getattr(skdata, name)().astype(np.float32) / 255.0
            except Exception:  # noqa: BLE001 - dataset not bundled in this install
                continue
    except ImportError:
        pass
    if extra_dir is not None and extra_dir.is_dir():
        from app.pipeline.io import decode_image  # noqa: PLC0415

        for p in sorted(extra_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                refs[p.stem] = decode_image(p.read_bytes(), 200_000_000).rgb
    out = {}
    for k, v in refs.items():
        h, w = v.shape[:2]
        s = min(1.0, max_side / max(h, w))
        out[k] = v if s == 1.0 else cv2.resize(v, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return out


def load_lol_v2_pairs(root: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """LOL-v2-real test pairs from `<root>/Test/Low` and `<root>/Test/Normal`.

    Accepts either the Real_captured directory or its parent. LOL-v2 names pairs
    lowXXXXX.png / normalXXXXX.png.
    """
    from app.pipeline.io import decode_image  # noqa: PLC0415

    for cand in (root, root / "Real_captured"):
        low_dir, gt_dir = cand / "Test" / "Low", cand / "Test" / "Normal"
        if low_dir.is_dir() and gt_dir.is_dir():
            break
    else:
        raise FileNotFoundError(f"LOL-v2 Test/Low and Test/Normal not found under {root}")
    pairs = []
    for lp in sorted(low_dir.glob("*.png")):
        gp = gt_dir / lp.name.replace("low", "normal")
        if not gp.exists():
            gp = gt_dir / lp.name
        if gp.exists():
            pairs.append(
                (lp.stem, decode_image(lp.read_bytes(), 200_000_000).rgb, decode_image(gp.read_bytes(), 200_000_000).rgb)
            )
    return pairs
