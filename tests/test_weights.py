from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from app.errors import WeightsError
from app.weights import PINS_FILE, WeightSpec, load_manifest, verify_file


def _spec(sha: str | None, size: int | None = None) -> WeightSpec:
    return WeightSpec(name="toy", backend="x", file="toy/w.bin", url="http://invalid/", source="", sha256=sha, size=size)


def _write(tmp: Path, data: bytes) -> None:
    (tmp / "toy").mkdir(parents=True, exist_ok=True)
    (tmp / "toy" / "w.bin").write_bytes(data)


def test_manifest_parses_and_has_pinned_models():
    m = load_manifest(PROJECT_ROOT / "weights.yaml")
    for name in ("retinexformer_fivek", "retinexformer_lolv2_real", "zerodce_pp", "scunet_color_real_psnr"):
        assert name in m
    for name in ("retinexformer_fivek", "retinexformer_lolv2_real", "zerodce_pp"):
        assert m[name].sha256 and len(m[name].sha256) == 64
    assert m["retinexformer_fivek"].arch["n_feat"] == 40


def test_verify_ok(tmp_path):
    data = b"weights-bytes"
    _write(tmp_path, data)
    verify_file(_spec(hashlib.sha256(data).hexdigest(), len(data)), tmp_path, allow_unpinned=False)


def test_verify_checksum_mismatch_fails_clearly(tmp_path):
    _write(tmp_path, b"tampered")
    with pytest.raises(WeightsError, match="CHECKSUM MISMATCH"):
        verify_file(_spec("0" * 64), tmp_path, allow_unpinned=False)


def test_verify_size_mismatch(tmp_path):
    _write(tmp_path, b"short")
    with pytest.raises(WeightsError, match="size"):
        verify_file(_spec("0" * 64, size=999), tmp_path, allow_unpinned=False)


def test_missing_file(tmp_path):
    with pytest.raises(WeightsError, match="missing"):
        verify_file(_spec("0" * 64), tmp_path, allow_unpinned=False)


def test_unpinned_rejected_then_pinned_on_first_use(tmp_path):
    data = b"first-version"
    _write(tmp_path, data)
    with pytest.raises(WeightsError, match="no pinned sha256"):
        verify_file(_spec(None), tmp_path, allow_unpinned=False)
    verify_file(_spec(None), tmp_path, allow_unpinned=True)
    pins = json.loads((tmp_path / PINS_FILE).read_text())
    assert pins["toy"] == hashlib.sha256(data).hexdigest()
    # A later swap of the file is detected against the recorded pin.
    _write(tmp_path, b"swapped-version")
    with pytest.raises(WeightsError, match="CHECKSUM MISMATCH"):
        verify_file(_spec(None), tmp_path, allow_unpinned=True)


def test_malformed_sha_in_manifest(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text("weights:\n  a:\n    backend: x\n    file: f\n    url: u\n    sha256: nothex\n")
    with pytest.raises(WeightsError):
        load_manifest(p)
