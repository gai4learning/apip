"""Normalize safe usage data and CUHK/backend response headers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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


def summarize_headers(headers: Mapping[str, str]) -> HeaderSummary:
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    routing = {name: lowered[name] for name in ROUTING_HEADER_NAMES if name in lowered}
    cuhk = {name: lowered[name] for name in CUHK_HEADER_NAMES if name in lowered}
    backend = {
        name: value
        for name, value in lowered.items()
        if name.startswith("x-ratelimit-")
    }
    return HeaderSummary(routing=routing, cuhk_allowance=cuhk, backend_capacity=backend)


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
            value = usage.get(candidate)
            if isinstance(value, int):
                normalized[target] = value
                break

    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    if isinstance(details, Mapping) and isinstance(details.get("reasoning_tokens"), int):
        normalized["reasoning_tokens"] = details["reasoning_tokens"]
    return normalized


def safe_session_export(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    allowed = {
        "operation", "region", "requested_model", "served_model", "status_code",
        "finish_reason", "latency_ms", "prompt_tokens", "completion_tokens",
        "reasoning_tokens", "total_tokens", "image_size", "quality",
        "output_format", "count", "vector_count", "vector_dimension",
        "request_id", "apim_request_id", "timestamp_utc",
    }
    safe_records = [
        {key: value for key, value in record.items() if key in allowed}
        for record in records
    ]
    return {
        "scope": "Application-side statistics for this local Streamlit session only",
        "records": safe_records,
    }
