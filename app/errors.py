from __future__ import annotations

from typing import Any


class APIError(Exception):
    """Error that maps directly to a structured JSON HTTP response."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidImageError(APIError):
    status_code = 400
    code = "invalid_image"


class InvalidParameterError(APIError):
    status_code = 422
    code = "invalid_parameter"


class UnsupportedMediaTypeError(APIError):
    status_code = 415
    code = "unsupported_media_type"


class PayloadTooLargeError(APIError):
    status_code = 413
    code = "payload_too_large"


class BackendUnavailableError(APIError):
    status_code = 503
    code = "backend_unavailable"


class RequestTimeoutError(APIError):
    status_code = 504
    code = "timeout"


class WeightsError(RuntimeError):
    """Raised for missing, unpinned or checksum-mismatched weight files."""
