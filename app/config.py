from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All runtime configuration. Every field maps to an env var prefixed with LLE_.

    List values are given as JSON, e.g. LLE_ENABLED_BACKENDS='["retinexformer","classical"]'.
    """

    model_config = SettingsConfigDict(env_prefix="LLE_", env_file=".env", extra="ignore")

    # --- weights / models -------------------------------------------------
    weights_dir: Path = Path("/weights")
    weights_manifest: Path = PROJECT_ROOT / "weights.yaml"
    verify_weights: bool = True
    allow_unpinned_weights: bool = False
    device: str = "auto"  # auto | cpu | cuda | cuda:N
    torch_threads: int = Field(0, ge=0, description="0 = torch default")
    enabled_backends: list[Literal["retinexformer", "zerodce", "classical"]] = [
        "retinexformer",
        "zerodce",
        "classical",
    ]
    required_backends: list[str] = Field(
        default_factory=list,
        description="Backends (or checkpoint names) whose load failure aborts startup.",
    )
    default_method: Literal["auto", "retinexformer", "zerodce", "classical"] = "auto"
    auto_candidates: list[str] = [
        "retinexformer:retinexformer_lolv2_real:pc",
        "retinexformer:retinexformer_lolv2_real:nopc",
        "retinexformer:retinexformer_fivek:pc",
        "retinexformer:retinexformer_fivek:nopc",
        "zerodce:zerodce_pp:pc",
        "classical::pc",
    ]
    auto_preview_side: int = Field(512, ge=128)
    auto_weights: dict[str, float] = Field(default_factory=dict, description="overrides of auto score weights")
    auto_use_niqe: bool = True
    auto_strengths: list[float] = [0.4, 0.55, 0.7, 0.85, 1.0]
    pre_gain_targets: dict[str, float] = {
        "retinexformer_lolv2_real": 0.10,
        "retinexformer_fivek": 0.22,
        "zerodce": 0.12,
        "classical": 0.18,
    }
    pre_gain_max: float = Field(1024.0, ge=1.0, description="cap on the linear-light pre-exposure gain")
    ae_white_target: float | None = Field(0.93, gt=0.0, le=1.0, description="seamless background level; null disables")
    ae_bg_min_fraction: float = Field(0.2, gt=0.0, le=1.0)
    ae_white_max_input: float = Field(0.7, gt=0.0, le=1.0)
    shadow_protect: bool = True  # keep intentional blacks (night sky, black backdrop) black
    shadow_max_level: float = Field(0.08, gt=0.0, le=0.5)
    shadow_rel_level: float = Field(0.3, gt=0.0, le=1.0)
    shadow_min_area: float = Field(0.10, ge=0.0, le=1.0, description="min share of the frame of one black region")
    shadow_max_texture: float = Field(0.0035, gt=0.0)
    chroma_suppress: bool = True  # smooth color noise where the local gain is high
    chroma_gain_low: float = Field(2.0, ge=1.0)
    chroma_gain_high: float = Field(5.0, ge=1.0)
    chroma_radius: int = Field(8, ge=1)
    retinexformer_checkpoints: list[str] = ["retinexformer_fivek", "retinexformer_lolv2_real"]
    retinexformer_default_checkpoint: str = "retinexformer_fivek"
    zerodce_checkpoint: str = "zerodce_pp"
    scunet_checkpoint: str = "scunet_color_real_psnr"
    enable_scunet: bool = True

    # --- input limits -----------------------------------------------------
    max_input_megapixels: float = Field(24.0, gt=0)
    max_upload_mb: float = Field(60.0, gt=0)
    max_batch_files: int = Field(20, ge=1)

    # --- large-image handling ---------------------------------------------
    enhance_max_model_pixels: int = Field(
        250_000, ge=4096, description="Retinexformer runs at most at this many pixels (~1 GB RAM per 0.2 MP on CPU)."
    )
    large_image_mode: Literal["guided", "tile"] = "guided"
    tile_size: int = Field(512, ge=64)
    tile_overlap: int = Field(64, ge=8)
    guided_upsample_radius: int = Field(4, ge=1)
    guided_upsample_eps: float = Field(1e-4, gt=0)
    zerodce_max_side: int = Field(512, ge=64)

    # --- denoising --------------------------------------------------------
    denoise: Literal["auto", "on", "off"] = "auto"
    denoise_backend: Literal["auto", "scunet", "opencv"] = "auto"
    denoise_strength: float = Field(1.0, ge=0.0, le=1.0)
    denoise_tile_size: int = Field(512, ge=64)
    denoise_tile_overlap: int = Field(32, ge=8)
    dark_threshold: float = Field(0.30, ge=0.0, le=1.0)
    noise_threshold: float = Field(0.02, ge=0.0)

    # --- highlight protection ----------------------------------------------
    highlight_low: float = Field(0.6, ge=0.0, le=1.0)
    highlight_high: float = Field(0.9, ge=0.0, le=1.0)
    highlight_mask_radius: int = Field(8, ge=1)
    highlight_mask_eps: float = Field(1e-3, gt=0)
    knee: float = Field(0.85, gt=0.0, lt=0.99)
    clip_tolerance: float = Field(0.002, ge=0.0, le=1.0, description="fraction, 0.002 = 0.2%")
    preserve_color: bool = True  # used when a method is forced and preserve_color is not given
    max_gain: float = Field(48.0, ge=1.0)

    # --- final adjustments ------------------------------------------------
    ae_target_low: float = Field(0.40, ge=0.0, le=1.0)
    ae_target_high: float = Field(0.50, ge=0.0, le=1.0)
    ae_max_gain: float = Field(4.0, ge=1.0)
    ae_allow_darken: bool = False
    sharpen: bool = False
    sharpen_amount: float = Field(0.3, ge=0.0, le=1.0)
    sharpen_sigma: float = Field(1.0, gt=0.0, le=5.0)
    clahe_clip_limit: float = Field(2.0, gt=0.0)
    clahe_grid: int = Field(8, ge=1)
    classical_target: float = Field(0.45, gt=0.0, lt=1.0)
    strength: float | None = Field(None, ge=0.0, le=1.0, description="null = chosen per image")

    # --- output -----------------------------------------------------------
    output_format: Literal["jpeg", "png", "webp"] = "jpeg"
    jpeg_quality: int = Field(95, ge=1, le=100)
    webp_quality: int = Field(95, ge=1, le=100)
    strip_exif: bool = True

    # --- metrics ----------------------------------------------------------
    iqa_enabled: bool = True
    iqa_max_side: int = Field(1024, ge=64)

    # --- service ----------------------------------------------------------
    max_concurrency: int = Field(1, ge=1)
    request_timeout_s: float = Field(180.0, gt=0)
    log_level: str = "INFO"
    log_json: bool = True

    @model_validator(mode="after")
    def _check_ranges(self) -> "Settings":
        if self.highlight_low >= self.highlight_high:
            raise ValueError("highlight_low must be < highlight_high")
        if self.ae_target_low > self.ae_target_high:
            raise ValueError("ae_target_low must be <= ae_target_high")
        if self.tile_overlap * 2 >= self.tile_size:
            raise ValueError("tile_overlap must be < tile_size / 2")
        if self.denoise_tile_overlap * 2 >= self.denoise_tile_size:
            raise ValueError("denoise_tile_overlap must be < denoise_tile_size / 2")
        if not self.auto_strengths or not all(0.0 <= s <= 1.0 for s in self.auto_strengths):
            raise ValueError("auto_strengths must be a non-empty list of values in [0, 1]")
        return self

    @property
    def max_input_pixels(self) -> int:
        return int(self.max_input_megapixels * 1_000_000)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
