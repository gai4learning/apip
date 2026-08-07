from __future__ import annotations

import json
from typing import Any

import pytest

from clients.cuhk_apim_client import CUHKAPIMClient


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        body: Any | None = None,
        headers: dict[str, str] | None = None,
        raw_content: bytes | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = {} if body is None else body
        self.headers = headers or {}
        self.content = raw_content if raw_content is not None else json.dumps(self._body).encode()
        self._json_error = json_error

    def json(self) -> Any:
        if self._json_error:
            raise self._json_error
        return self._body


class RecordingHTTPClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


@pytest.fixture
def client_factory():
    created: list[RecordingHTTPClient] = []

    def factory(
        response: FakeResponse,
        base_url: str = "https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1",
        api_key: str = "local-test-subscription-key",
    ) -> tuple[CUHKAPIMClient, RecordingHTTPClient]:
        transport = RecordingHTTPClient(response)
        created.append(transport)
        return CUHKAPIMClient(base_url, api_key, http_client=transport), transport

    return factory
