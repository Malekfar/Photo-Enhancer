from __future__ import annotations

import base64
import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.service import ModelRegistry
from tests.conftest import make_settings


def _png(h=96, w=128, level=0.15, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = (np.clip(level * (0.5 + rng.random((h, w, 3))), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(settings, registry):
    with TestClient(create_app(settings, registry)) as c:
        yield c


@pytest.fixture(scope="module")
def small_registry():
    s = make_settings(enabled_backends=["classical"], enable_scunet=False, max_input_megapixels=0.005)
    reg = ModelRegistry(s)
    reg.load()
    return reg


def _files(data: bytes, name="a.png", ctype="image/png"):
    return {"file": (name, data, ctype)}


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "before" in r.text


def test_health_and_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    m = client.get("/v1/models").json()
    assert "classical" in m["enhancers"]
    assert "opencv" in m["denoisers"]


def test_auto_method_reports_its_decision(client):
    r = client.post("/v1/enhance/json", files=_files(_png(level=0.02)), data={"method": "auto"})
    assert r.status_code == 200, r.text
    pipe = r.json()["pipeline"]
    assert pipe["auto"]["chosen"] == pipe["auto"]["candidates"][0]["candidate"]
    assert pipe["resolved"]["method"] in ("retinexformer", "zerodce", "classical")
    assert pipe["resolved"]["preserve_color"] in (True, False)
    assert 0.0 <= pipe["resolved"]["strength"] <= 1.0
    assert pipe["auto"]["strength"]["chosen"] == pipe["resolved"]["strength"]
    assert pipe["pre_exposure"]["gain"] >= 1.0


def test_enhance_returns_image_with_metrics(client):
    r = client.post("/v1/enhance", files=_files(_png()), data={"method": "classical"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/jpeg"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (128, 96)
    metrics = json.loads(r.headers["x-enhance-metrics"])
    assert metrics["after"]["median_luminance"] > metrics["before"]["median_luminance"]
    assert "clipped_pct" in metrics["before"]
    assert r.headers.get("x-request-id")


@pytest.mark.parametrize("fmt,ctype", [("png", "image/png"), ("webp", "image/webp"), ("jpeg", "image/jpeg")])
def test_output_formats(client, fmt, ctype):
    r = client.post("/v1/enhance", files=_files(_png()), data={"method": "classical", "output_format": fmt})
    assert r.status_code == 200
    assert r.headers["content-type"] == ctype


def test_enhance_json(client):
    r = client.post(
        "/v1/enhance/json",
        files=_files(_png()),
        data={"method": "classical", "strength": "0.8", "preserve_color": "true", "denoise": "off"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    img = Image.open(io.BytesIO(base64.b64decode(body["image_base64"])))
    assert img.size == (body["width"], body["height"]) == (128, 96)
    assert body["params"]["strength"] == 0.8
    assert body["pipeline"]["denoise"]["applied"] is False
    for k in ("mean_luminance", "median_luminance", "clipped_pct"):
        assert k in body["metrics"]["before"] and k in body["metrics"]["after"]
    assert "total" in body["timings_ms"]


def test_iqa_metrics_when_available(client, registry):
    pytest.importorskip("pyiqa")
    r = client.post("/v1/enhance/json", files=_files(_png(192, 256)), data={"method": "classical", "iqa": "true"})
    assert r.status_code == 200
    after = r.json()["metrics"]["after"]
    assert "niqe" in after and "brisque" in after
    if after.get("iqa_error"):
        pytest.skip(f"pyiqa present but metrics unavailable: {after['iqa_error']}")
    assert after["niqe"] is not None and after["brisque"] is not None


def test_batch_zip_with_partial_failure(client):
    files = [
        ("files", ("one.png", _png(seed=1), "image/png")),
        ("files", ("two.png", _png(seed=2), "image/png")),
        ("files", ("bad.png", b"not an image", "image/png")),
        ("files", ("one.png", _png(seed=3), "image/png")),
    ]
    r = client.post("/v1/enhance/batch", files=files, data={"method": "classical", "output_format": "png"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert {"one_enhanced.png", "two_enhanced.png", "one_enhanced_1.png", "metrics.json"} <= names
    report = json.loads(zf.read("metrics.json"))
    assert len(report["results"]) == 3
    assert report["errors"][0]["file"] == "bad.png"
    assert r.headers["x-batch-failed"] == "1"


def test_batch_all_failed(client):
    r = client.post("/v1/enhance/batch", files=[("files", ("x.png", b"junk", "image/png"))])
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "batch_failed"


def test_batch_too_many_files(client, settings):
    files = [("files", (f"{i}.png", b"x", "image/png")) for i in range(settings.max_batch_files + 1)]
    r = client.post("/v1/enhance/batch", files=files)
    assert r.status_code == 413


def test_unsupported_and_invalid_inputs(client):
    r = client.post("/v1/enhance", files=_files(b"hello", "a.txt", "text/plain"))
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"
    r = client.post("/v1/enhance", files=_files(b"garbage-bytes"))
    assert r.status_code == 415
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(buf, format="GIF")
    r = client.post("/v1/enhance", files=_files(buf.getvalue(), "a.gif", "application/octet-stream"))
    assert r.status_code == 415


@pytest.mark.parametrize(
    "data",
    [
        {"strength": "2"},
        {"strength": "abc"},
        {"highlight_low": "0.9", "highlight_high": "0.5"},
        {"method": "magic"},
        {"output_format": "tiff"},
        {"preserve_color": "maybe"},
        {"denoise": "sometimes"},
    ],
)
def test_invalid_params(client, data):
    r = client.post("/v1/enhance", files=_files(_png()), data=data)
    assert r.status_code == 422
    assert r.json()["error"]["code"] in ("invalid_parameter", "invalid_request")


def test_missing_file(client):
    r = client.post("/v1/enhance", data={"method": "classical"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_upload_size_limit(settings, registry):
    app = create_app(make_settings(max_upload_mb=0.001), registry)
    with TestClient(app) as c:
        r = c.post("/v1/enhance", files=_files(_png(256, 256, seed=5)))
    assert r.status_code == 413


def test_pixel_limit_and_unavailable_backend(small_registry):
    with TestClient(create_app(small_registry.settings, small_registry)) as c:
        r = c.post("/v1/enhance", files=_files(_png(96, 128)))
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "payload_too_large"
        r = c.post("/v1/enhance", files=_files(_png(40, 40)), data={"method": "zerodce"})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "backend_unavailable"


def test_timeout(settings, registry):
    app = create_app(make_settings(request_timeout_s=0.0001), registry)
    with TestClient(app) as c:
        r = c.post("/v1/enhance", files=_files(_png(512, 512)), data={"method": "classical"})
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "timeout"


def test_request_id_propagated(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
