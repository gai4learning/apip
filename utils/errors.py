"""Structured, credential-safe API errors and user guidance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from utils.security import redact_text
from utils.usage import HeaderSummary


@dataclass(slots=True)
class APIMError(Exception):
    status_code: int | None
    code: str
    guidance: str
    headers: HeaderSummary | None = None

    def __str__(self) -> str:
        return self.guidance


class ResponseFormatError(APIMError):
    pass


def error_code(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return "unknown_error"
    error = payload.get("error")
    if isinstance(error, Mapping):
        code = error.get("code") or error.get("type")
        message = error.get("message")
        message_text = message.lower() if isinstance(message, str) else ""
        if "max_tokens" in message_text:
            return "max_tokens_unsupported"
        if "unknown_model" in message_text:
            return "unknown_model"
        if isinstance(code, str):
            return code
    return "unknown_error"


def guidance_for_status(
    status_code: int,
    code: str = "unknown_error",
    retry_after: str | None = None,
) -> str:
    normalized = code.lower()
    if status_code == 400 and "max_tokens" in normalized:
        return "This modern Chat Completions deployment requires max_completion_tokens, not max_tokens."
    if status_code == 400 and ("unknown_model" in normalized or "model" in normalized):
        return "The model ID is not deployed in the selected regional API. Choose a model listed for that region and operation."
    if status_code == 401:
        return "The CUHK APIM subscription key is missing or invalid. Check the local key configuration; the key is not shown here."
    if status_code == 403:
        return "Access was denied. A call quota, token quota, product authorization, or another access condition may have been reached."
    if status_code == 404:
        return "The operation was not found. Check the selected region, OpenAI v1 operation path, and whether APIM exposes this specialized operation."
    if status_code == 429:
        suffix = f" Retry after {retry_after}." if retry_after else ""
        return (
            "The request was rate limited. It may be the CUHK APIM call-rate limit, "
            "CUHK APIM token-rate limit, or Foundry backend deployment capacity."
            + suffix
        )
    return f"The CUHK APIM request failed with HTTP {status_code}. Use the sanitized request IDs when seeking support."


def sanitized_unexpected_error(error: Exception, api_key: str | None = None) -> str:
    text = redact_text(error, (api_key or "",))
    lowered = text.lower()
    if "api-key" in lowered or "authorization" in lowered or "bearer" in lowered:
        return "The request failed. Sensitive credential details were removed; rotate the key if exposure is suspected."
    return "The request failed before a valid CUHK APIM response was available. Check local configuration and try again."
