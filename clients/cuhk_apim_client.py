"""Minimal OpenAI-v1 HTTP client for CUHK APIM regional APIs."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from utils.errors import APIMError, ResponseFormatError, error_code, guidance_for_status
from utils.security import require_api_key
from utils.usage import HeaderSummary, summarize_headers

DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_RESPONSE_BYTES = 40 * 1024 * 1024


class HTTPClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: float,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class APIMResponse:
    status_code: int
    body: dict[str, Any]
    headers: HeaderSummary
    latency_ms: int


@dataclass(frozen=True, slots=True)
class _BufferedResponse:
    status_code: int
    headers: Mapping[str, str]
    content: bytes


class CUHKAPIMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        http_client: HTTPClient | None = None,
        logger: logging.Logger | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        try:
            self._api_key = require_api_key(api_key)
        except ValueError as error:
            raise APIMError(401, "missing_subscription_key", str(error)) from None
        if http_client is None:
            import httpx

            self._http = httpx.Client(follow_redirects=False)
            self._stream_responses = True
        else:
            self._http = http_client
            self._stream_responses = False
        self._logger = logger
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes

    def operation_url(self, operation: str) -> str:
        return f"{self.base_url}/{operation.lstrip('/')}"

    def post(self, operation: str, payload: Mapping[str, Any]) -> APIMResponse:
        url = self.operation_url(operation)
        headers = {
            "api-key": self._api_key,
            "content-type": "application/json",
            "accept": "application/json",
        }
        started = time.perf_counter()
        try:
            response = self._send(url, headers, payload)
        except APIMError:
            raise
        except Exception:  # noqa: BLE001 - transport implementations vary
            if self._logger:
                self._logger.warning("APIM transport failure operation=%s", operation)
            raise APIMError(
                None,
                "transport_error",
                "The request could not reach CUHK APIM. Check the Codespace network and regional base URL.",
            ) from None

        latency_ms = round((time.perf_counter() - started) * 1000)
        header_summary = summarize_headers(response.headers)
        status_code = response.status_code
        try:
            body = json.loads(response.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if not 200 <= status_code < 300:
                raise APIMError(
                    status_code,
                    "invalid_error_response",
                    guidance_for_status(
                        status_code,
                        retry_after=header_summary.cuhk_allowance.get("retry-after"),
                    ),
                    header_summary,
                ) from None
            raise ResponseFormatError(
                status_code,
                "malformed_json",
                "CUHK APIM returned a response that was not valid JSON.",
                header_summary,
            ) from None
        if not isinstance(body, dict):
            raise ResponseFormatError(
                status_code,
                "unexpected_json",
                "CUHK APIM returned an unexpected JSON response shape.",
                header_summary,
            )

        if not 200 <= status_code < 300:
            code = error_code(body)
            retry_after = header_summary.cuhk_allowance.get("retry-after")
            raise APIMError(
                status_code,
                code,
                guidance_for_status(status_code, code, retry_after),
                header_summary,
            )
        if self._logger:
            self._logger.info(
                "APIM request completed operation=%s status=%s latency_ms=%s",
                operation,
                status_code,
                latency_ms,
            )
        return APIMResponse(status_code, body, header_summary, latency_ms)

    def _send(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> _BufferedResponse:
        if self._stream_responses:
            with self._http.stream(
                "POST",
                url,
                headers=headers,
                json=dict(payload),
                timeout=self._timeout,
            ) as response:
                response_headers = dict(response.headers)
                self._reject_declared_oversize(response.status_code, response_headers)
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > self._max_response_bytes:
                        raise self._oversize_error(response.status_code, response_headers)
                return _BufferedResponse(
                    int(response.status_code), response_headers, bytes(content)
                )

        response = self._http.post(
            url,
            headers=headers,
            json=dict(payload),
            timeout=self._timeout,
        )
        response_headers = dict(getattr(response, "headers", {}))
        status_code = int(response.status_code)
        self._reject_declared_oversize(status_code, response_headers)
        content = bytes(getattr(response, "content", b""))
        if len(content) > self._max_response_bytes:
            raise self._oversize_error(status_code, response_headers)
        return _BufferedResponse(status_code, response_headers, content)

    def _reject_declared_oversize(
        self, status_code: int, headers: Mapping[str, str]
    ) -> None:
        lowered = {str(key).lower(): str(value) for key, value in headers.items()}
        value = lowered.get("content-length")
        try:
            declared_length = int(value) if value else 0
        except ValueError:
            declared_length = 0
        if declared_length > self._max_response_bytes:
            raise self._oversize_error(status_code, headers)

    @staticmethod
    def _oversize_error(
        status_code: int, headers: Mapping[str, str]
    ) -> ResponseFormatError:
        return ResponseFormatError(
            status_code,
            "response_too_large",
            "The API response exceeded the safe local response-size limit.",
            summarize_headers(headers),
        )
