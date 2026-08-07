"""Secret redaction and safe local logging helpers."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_REDACTED = "[REDACTED]"
_HEADER_SECRET = re.compile(
    r"(?i)(api[-_]?key|authorization|subscription[-_]?key)(\s*[:=]\s*)([^,;\]}\r\n]+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_SECRET_FIELD_NAMES = {
    "api-key", "api_key", "authorization", "subscription_key", "key",
    "prompt", "messages", "input", "b64_json", "embedding", "vector", "vectors",
    "source_file", "source_files", "response_body", "request_body",
}


def require_api_key(api_key: str | None) -> str:
    key = (api_key or "").strip()
    if not key:
        raise ValueError(
            "The CUHK APIM subscription key is missing. Set CUHK_APIM_API_KEY or enter it in the password-masked field."
        )
    return key


def mask_api_key(api_key: str | None) -> str:
    return "Configured" if (api_key or "").strip() else "Not configured"


def redact_text(value: object, secrets: tuple[str, ...] = ()) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, _REDACTED)
    text = _HEADER_SECRET.sub(lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}", text)
    return _BEARER_SECRET.sub(f"Bearer {_REDACTED}", text)


def sanitize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        normalized = key.lower().replace(" ", "_")
        if normalized in _SECRET_FIELD_NAMES:
            continue
        if isinstance(item, Mapping):
            sanitized[key] = sanitize_mapping(item)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            sanitized[key] = redact_text(item)
    return sanitized


def configure_local_logging(log_dir: str = "logs") -> logging.Logger:
    """Create metadata-only rotating logs; prompts and response bodies are never logged."""
    logger = logging.getLogger("cuhk_apip_demo")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if _env_flag("CUHK_ENABLE_LOCAL_DEBUG_LOGGING") else logging.INFO)
    logger.propagate = False
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path / "cuhk_apip_demo.log",
        maxBytes=512_000,
        backupCount=2,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
