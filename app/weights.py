"""Weights manifest handling: parsing, checksum verification, download with resume."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.errors import WeightsError

log = logging.getLogger(__name__)

PINS_FILE = "pins.json"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class WeightSpec:
    name: str
    backend: str
    file: str
    url: str
    source: str
    sha256: str | None
    size: int | None
    description: str = ""
    arch: dict[str, Any] = field(default_factory=dict)


def load_manifest(path: Path) -> dict[str, WeightSpec]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    specs: dict[str, WeightSpec] = {}
    for name, entry in (raw.get("weights") or {}).items():
        sha = (entry.get("sha256") or "").strip().lower() or None
        if sha is not None and (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise WeightsError(f"weights.yaml: '{name}' has a malformed sha256")
        specs[name] = WeightSpec(
            name=name,
            backend=entry["backend"],
            file=entry["file"],
            url=entry["url"],
            source=entry.get("source", ""),
            sha256=sha,
            size=entry.get("size"),
            description=entry.get("description", ""),
            arch=dict(entry.get("arch") or {}),
        )
    return specs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pins(weights_dir: Path) -> dict[str, str]:
    p = weights_dir / PINS_FILE
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_pin(weights_dir: Path, name: str, digest: str) -> None:
    pins = _load_pins(weights_dir)
    pins[name] = digest
    tmp = weights_dir / (PINS_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(pins, fh, indent=2, sort_keys=True)
    os.replace(tmp, weights_dir / PINS_FILE)


def expected_sha256(spec: WeightSpec, weights_dir: Path) -> tuple[str | None, str]:
    """Return (digest, origin) where origin is 'manifest', 'pin' or 'none'."""
    if spec.sha256:
        return spec.sha256, "manifest"
    pinned = _load_pins(weights_dir).get(spec.name)
    if pinned:
        return pinned, "pin"
    return None, "none"


def verify_file(spec: WeightSpec, weights_dir: Path, *, allow_unpinned: bool) -> Path:
    """Validate an existing weight file. Returns its path or raises WeightsError."""
    path = weights_dir / spec.file
    if not path.is_file():
        raise WeightsError(
            f"weights '{spec.name}' missing at {path}. Run: python -m scripts.download_weights"
        )
    if spec.size is not None and path.stat().st_size != spec.size:
        raise WeightsError(
            f"weights '{spec.name}': size {path.stat().st_size} != expected {spec.size} "
            f"(incomplete or wrong file at {path}). Delete it and re-download."
        )
    digest = sha256_file(path)
    expected, origin = expected_sha256(spec, weights_dir)
    if expected is None:
        if not allow_unpinned:
            raise WeightsError(
                f"weights '{spec.name}' has no pinned sha256 in weights.yaml. "
                f"Set LLE_ALLOW_UNPINNED_WEIGHTS=true to pin it on first use (sha256={digest})."
            )
        _save_pin(weights_dir, spec.name, digest)
        log.warning(
            "pinned unpinned weights on first use; copy this sha256 into weights.yaml",
            extra={"weights": spec.name, "sha256": digest},
        )
        return path
    if digest != expected:
        raise WeightsError(
            f"CHECKSUM MISMATCH for weights '{spec.name}' at {path}: "
            f"got {digest}, expected {expected} (from {origin}). Refusing to load."
        )
    return path


def _open(url: str, offset: int) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "lowlight-enhancer/1.0"})
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    return urllib.request.urlopen(req, timeout=60)


def download(
    spec: WeightSpec,
    weights_dir: Path,
    *,
    allow_unpinned: bool,
    retries: int = 20,
) -> Path:
    """Download (resumable) and verify. Existing valid files are not re-downloaded."""
    dest = weights_dir / spec.file
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected, _ = expected_sha256(spec, weights_dir)
    if expected is None and not allow_unpinned:
        raise WeightsError(
            f"weights '{spec.name}' is not pinned in weights.yaml; refusing to download. "
            "Set LLE_ALLOW_UNPINNED_WEIGHTS=true to download and pin on first use."
        )

    if dest.is_file():
        try:
            return verify_file(spec, weights_dir, allow_unpinned=allow_unpinned)
        except WeightsError as exc:
            log.warning("existing weights invalid, re-downloading", extra={"error": str(exc)})
            dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while True:
        offset = part.stat().st_size if part.exists() else 0
        if spec.size is not None and offset >= spec.size:
            break
        try:
            with _open(spec.url, offset) as resp:
                ctype = resp.headers.get("Content-Type", "")
                if "text/html" in ctype:
                    raise WeightsError(
                        f"'{spec.name}': server returned an HTML page instead of the file "
                        f"(quota/confirmation page?). URL: {spec.url}"
                    )
                status = getattr(resp, "status", 200)
                mode = "ab" if offset and status == 206 else "wb"
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
            if spec.size is None:
                break
            if part.stat().st_size < spec.size:
                attempt += 1
                if attempt > retries:
                    raise WeightsError(f"download of '{spec.name}' kept ending early")
        except WeightsError:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            attempt += 1
            if attempt > retries:
                raise WeightsError(f"download of '{spec.name}' failed: {exc}") from exc
            wait = min(30, 2 ** min(attempt, 5))
            log.warning(
                "download interrupted, resuming",
                extra={"weights": spec.name, "attempt": attempt, "error": str(exc), "wait_s": wait},
            )
            time.sleep(wait)

    os.replace(part, dest)
    try:
        return verify_file(spec, weights_dir, allow_unpinned=allow_unpinned)
    except WeightsError:
        dest.unlink(missing_ok=True)
        raise
