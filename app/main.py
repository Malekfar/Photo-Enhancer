from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.config import Settings, get_settings
from app.errors import APIError, InvalidParameterError, PayloadTooLargeError, UnsupportedMediaTypeError
from app.logging_setup import request_id_var, setup_logging
from app.pipeline import EnhanceParams
from app.pipeline.io import EXTENSIONS
from app.service import InferenceRunner, ModelRegistry, ProcessedImage, process_image, to_jsonable

log = logging.getLogger("app")
STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "application/octet-stream", ""}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _error_body(code: str, message: str, details: dict | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or {}, "request_id": request_id_var.get()}}


def _parse_bool(name: str, value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise InvalidParameterError(f"'{name}' must be a boolean", details={name: value})


def _parse_float(name: str, value: str | None, default: float, lo: float, hi: float) -> float:
    if value is None or value == "":
        return default
    try:
        f = float(value)
    except ValueError as exc:
        raise InvalidParameterError(f"'{name}' must be a number", details={name: value}) from exc
    if not lo <= f <= hi:
        raise InvalidParameterError(f"'{name}' must be in [{lo}, {hi}]", details={name: f})
    return f


def _choice(name: str, value: str | None, default: str, options: set[str]) -> str:
    if value is None or value == "":
        return default
    v = value.strip().lower()
    if v not in options:
        raise InvalidParameterError(f"'{name}' must be one of {sorted(options)}", details={name: value})
    return v


def enhance_params(
    request: Request,
    method: Annotated[str | None, Form()] = None,
    checkpoint: Annotated[str | None, Form()] = None,
    strength: Annotated[str | None, Form()] = None,
    denoise: Annotated[str | None, Form()] = None,
    denoise_strength: Annotated[str | None, Form()] = None,
    highlight_low: Annotated[str | None, Form()] = None,
    highlight_high: Annotated[str | None, Form()] = None,
    preserve_color: Annotated[str | None, Form()] = None,
    sharpen: Annotated[str | None, Form()] = None,
    output_format: Annotated[str | None, Form()] = None,
    quality: Annotated[str | None, Form()] = None,
    strip_exif: Annotated[str | None, Form()] = None,
    iqa: Annotated[str | None, Form()] = None,
) -> EnhanceParams:
    s: Settings = request.app.state.settings
    dn = (denoise or "").strip().lower()
    dn = {"true": "on", "1": "on", "yes": "on", "false": "off", "0": "off", "no": "off"}.get(dn, dn)
    fmt = _choice("output_format", output_format, s.output_format, {"jpeg", "jpg", "png", "webp"})
    fmt = "jpeg" if fmt == "jpg" else fmt
    default_q = s.webp_quality if fmt == "webp" else s.jpeg_quality
    lo = _parse_float("highlight_low", highlight_low, s.highlight_low, 0.0, 1.0)
    hi = _parse_float("highlight_high", highlight_high, s.highlight_high, 0.0, 1.0)
    if lo >= hi:
        raise InvalidParameterError("highlight_low must be < highlight_high", details={"highlight_low": lo, "highlight_high": hi})
    return EnhanceParams(
        method=_choice("method", method, s.default_method, {"auto", "retinexformer", "zerodce", "classical"}),  # type: ignore[arg-type]
        checkpoint=(checkpoint or None),
        strength=(
            s.strength
            if (strength or "").strip().lower() in ("", "auto")
            else _parse_float("strength", strength, 1.0, 0.0, 1.0)
        ),
        denoise=_choice("denoise", dn, s.denoise, {"auto", "on", "off"}),  # type: ignore[arg-type]
        denoise_strength=_parse_float("denoise_strength", denoise_strength, s.denoise_strength, 0.0, 1.0),
        highlight_low=lo,
        highlight_high=hi,
        preserve_color=(
            None
            if (preserve_color or "").strip().lower() in ("", "auto")
            else _parse_bool("preserve_color", preserve_color, s.preserve_color)
        ),
        sharpen=_parse_bool("sharpen", sharpen, s.sharpen),
        output_format=fmt,  # type: ignore[arg-type]
        quality=int(_parse_float("quality", quality, default_q, 1, 100)),
        strip_exif=_parse_bool("strip_exif", strip_exif, s.strip_exif),
        iqa=_parse_bool("iqa", iqa, s.iqa_enabled),
    )


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if ctype not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedMediaTypeError(
            f"content type {ctype!r} not accepted; send JPEG, PNG or WebP", details={"filename": file.filename}
        )
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"file exceeds {max_bytes / 1024 / 1024:.0f} MB", details={"filename": file.filename}
        )
    return data


def _metrics_header(p: ProcessedImage) -> str:
    return json.dumps(to_jsonable(p.metrics), separators=(",", ":"))


def create_app(settings: Settings | None = None, registry: ModelRegistry | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        reg = registry or ModelRegistry(settings)
        if registry is None:
            t0 = time.perf_counter()
            await asyncio.to_thread(reg.load)
            log.info("models loaded", extra={"seconds": round(time.perf_counter() - t0, 2), **to_jsonable(reg.describe())})
        app.state.registry = reg
        app.state.runner = InferenceRunner(settings.max_concurrency, settings.request_timeout_s)
        yield
        app.state.runner.shutdown()

    app = FastAPI(
        title="Low-light product photo enhancer",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "duration_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                },
            )
            request_id_var.reset(token)

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError):
        if exc.status_code >= 500:
            log.error("api error", extra={"code": exc.code, "error": exc.message})
        return JSONResponse(_error_body(exc.code, exc.message, exc.details), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        return JSONResponse(
            _error_body("invalid_request", "request validation failed", {"errors": to_jsonable(exc.errors())}),
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(_error_body("http_error", str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(_error_body("internal_error", "internal server error"), status_code=500)

    async def _process(request: Request, data: bytes, params: EnhanceParams) -> ProcessedImage:
        reg: ModelRegistry = request.app.state.registry
        runner: InferenceRunner = request.app.state.runner
        return await runner.run(process_image, reg, data, params)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health(request: Request):
        reg: ModelRegistry | None = getattr(request.app.state, "registry", None)
        if reg is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        if settings.default_method == "auto":
            ok = bool(reg.auto_candidates(None, None))
        else:
            ok = settings.default_method in reg.enhancers
        body = {
            "status": "ok" if ok else "degraded",
            "version": __version__,
            "device": str(reg.device),
            "default_method": settings.default_method,
            "loaded_methods": sorted(reg.enhancers),
            "failures": reg.failures,
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/v1/models")
    async def models(request: Request):
        return to_jsonable(request.app.state.registry.describe())

    @app.post("/v1/enhance")
    async def enhance(
        request: Request,
        file: Annotated[UploadFile, File(description="JPEG/PNG/WebP image")],
        params: Annotated[EnhanceParams, Depends(enhance_params)],
    ):
        data = await read_upload(file, settings.max_upload_bytes)
        result = await _process(request, data, params)
        stem = Path(file.filename or "image").stem
        headers = {
            "X-Enhance-Metrics": _metrics_header(result),
            "X-Processing-Time-Ms": str(result.timings_ms.get("total")),
            "Content-Disposition": f'inline; filename="{stem}_enhanced.{EXTENSIONS[params.output_format]}"',
        }
        return Response(content=result.data, media_type=result.media_type, headers=headers)

    @app.post("/v1/enhance/json")
    async def enhance_json(
        request: Request,
        file: Annotated[UploadFile, File(description="JPEG/PNG/WebP image")],
        params: Annotated[EnhanceParams, Depends(enhance_params)],
    ):
        data = await read_upload(file, settings.max_upload_bytes)
        result = await _process(request, data, params)
        return JSONResponse(
            to_jsonable(
                {
                    "request_id": request_id_var.get(),
                    "media_type": result.media_type,
                    "width": result.width,
                    "height": result.height,
                    "image_base64": base64.b64encode(result.data).decode("ascii"),
                    "params": params.to_dict(),
                    "metrics": result.metrics,
                    "pipeline": result.pipeline,
                    "timings_ms": result.timings_ms,
                }
            )
        )

    @app.post("/v1/enhance/batch")
    async def enhance_batch(
        request: Request,
        files: Annotated[list[UploadFile], File(description="Multiple JPEG/PNG/WebP images")],
        params: Annotated[EnhanceParams, Depends(enhance_params)],
    ):
        if not files:
            raise InvalidParameterError("no files uploaded")
        if len(files) > settings.max_batch_files:
            raise PayloadTooLargeError(
                f"too many files: {len(files)} > {settings.max_batch_files}", details={"max_batch_files": settings.max_batch_files}
            )
        report: dict = {"params": params.to_dict(), "results": [], "errors": []}
        buf = io.BytesIO()
        used: set[str] = set()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            for idx, f in enumerate(files):
                name = Path(f.filename or f"image_{idx}").name
                try:
                    data = await read_upload(f, settings.max_upload_bytes)
                    result = await _process(request, data, params)
                except APIError as exc:
                    report["errors"].append({"file": name, "code": exc.code, "message": exc.message})
                    continue
                out_name = f"{Path(name).stem}_enhanced.{EXTENSIONS[params.output_format]}"
                n = 1
                while out_name in used:
                    out_name = f"{Path(name).stem}_enhanced_{n}.{EXTENSIONS[params.output_format]}"
                    n += 1
                used.add(out_name)
                zf.writestr(out_name, result.data)
                report["results"].append(
                    {"file": name, "output": out_name, "metrics": result.metrics, "pipeline": result.pipeline, "timings_ms": result.timings_ms}
                )
            zf.writestr("metrics.json", json.dumps(to_jsonable(report), indent=2))
        if not report["results"]:
            return JSONResponse(
                _error_body("batch_failed", "no file could be processed", {"errors": report["errors"]}), status_code=400
            )
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="enhanced.zip"',
                "X-Batch-Succeeded": str(len(report["results"])),
                "X-Batch-Failed": str(len(report["errors"])),
            },
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
