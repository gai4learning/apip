"""Normalize bounded usage metadata and credential-safe response headers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

MAX_METADATA_VALUE_CHARS = 256
MAX_BACKEND_HEADER_ENTRIES = 16
MAX_USAGE_VALUE = 1_000_000_000_000
MAX_EXPORT_RECORDS = 100

CUHK_HEADER_NAMES = (
    "x-cuhk-tokens-consumed",
    "x-cuhk-tpm-remaining",
    "x-cuhk-monthly-tokens-remaining",
    "retry-after",
)
ROUTING_HEADER_NAMES = (
    "x-ms-served-model",
    "x-ms-region",
    "x-ms-rai-invoked",
    "x-model-router-selected-model",
    "x-model-router-routing-mode",
    "x-model-router-fallback-occurred",
    "apim-request-id",
    "x-request-id",
)
STRING_EXPORT_FIELDS = {
    "operation",
    "region",
    "requested_model",
    "served_model",
    "finish_reason",
    "image_size",
    "quality",
    "output_format",
    "request_id",
    "apim_request_id",
    "timestamp_utc",
}
INTEGER_EXPORT_FIELDS = {
    "status_code",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "count",
    "vector_count",
    "vector_dimension",
}


@dataclass(frozen=True, slots=True)
class HeaderSummary:
    routing: dict[str, str] = field(default_factory=dict)
    cuhk_allowance: dict[str, str] = field(default_factory=dict)
    backend_capacity: dict[str, str] = field(default_factory=dict)

    @property
    def request_id(self) -> str | None:
        return self.routing.get("x-request-id")

    @property
    def apim_request_id(self) -> str | None:
        return self.routing.get("apim-request-id")


def safe_metadata_text(value: object) -> str:
    """Return bounded printable text for upstream/session metadata."""
    printable = "".join(
        character for character in str(value) if character >= " " and character != "\x7f"
    )
    return printable[:MAX_METADATA_VALUE_CHARS]


def summarize_headers(headers: Mapping[str, str]) -> HeaderSummary:
    lowered = {str(key).lower(): safe_metadata_text(value) for key, value in headers.items()}
    routing = {name: lowered[name] for name in ROUTING_HEADER_NAMES if name in lowered}
    cuhk = {name: lowered[name] for name in CUHK_HEADER_NAMES if name in lowered}
    backend: dict[str, str] = {}
    for name, value in lowered.items():
        if name.startswith("x-ratelimit-") and len(name) <= 64:
            backend[name] = value
            if len(backend) == MAX_BACKEND_HEADER_ENTRIES:
                break
    return HeaderSummary(routing=routing, cuhk_allowance=cuhk, backend_capacity=backend)


def _safe_usage_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_USAGE_VALUE:
        return value
    return None


def normalize_usage(data: Mapping[str, Any] | None) -> dict[str, int]:
    if not isinstance(data, Mapping):
        return {}
    usage = data.get("usage")
    if not isinstance(usage, Mapping):
        return {}

    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    normalized: dict[str, int] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            value = _safe_usage_integer(usage.get(candidate))
            if value is not None:
                normalized[target] = value
                break

    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    if isinstance(details, Mapping):
        reasoning = _safe_usage_integer(details.get("reasoning_tokens"))
        if reasoning is not None:
            normalized["reasoning_tokens"] = reasoning
    return normalized


def safe_session_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Allowlist and bound one record before it enters session state or an export."""
    safe: dict[str, Any] = {}
    for key in STRING_EXPORT_FIELDS:
        value = record.get(key)
        if isinstance(value, str):
            safe[key] = safe_metadata_text(value)
    for key in INTEGER_EXPORT_FIELDS:
        value = _safe_usage_integer(record.get(key))
        if value is not None:
            safe[key] = value
    return safe


def safe_session_export(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bounded_records = records[-MAX_EXPORT_RECORDS:]
    return {
        "scope": "Application-side statistics for this local Streamlit session only",
        "records": [safe_session_record(record) for record in bounded_records],
    }
