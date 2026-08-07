from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from conftest import FakeResponse

from clients.cuhk_apim_client import CUHKAPIMClient
from config.model_catalog import Region
from services.chat_service import ChatService
from utils.errors import APIMError, ResponseFormatError


def chat_body(content: str = "CUHK APIM test successful.", finish: str = "stop") -> dict:
    return {
        "id": "chat-1",
        "model": "served-version",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }


@pytest.mark.parametrize(
    ("region", "model_id", "base_url"),
    [
        (Region.EUS2, "gpt-5.4-mini", "https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1"),
        (Region.WUS3, "gpt-5.6-sol", "https://cuhk-apip.azure-api.net/foundry-wus3/openai/v1"),
    ],
)
def test_chat_url_api_key_and_modern_completion_field(
    client_factory, region: Region, model_id: str, base_url: str
) -> None:
    response = FakeResponse(200, chat_body(), {"x-ms-served-model": "served-header"})
    client, transport = client_factory(response, base_url)
    result = ChatService(client).complete(
        region=region,
        model_id=model_id,
        system_instruction="Be concise.",
        user_prompt="Test",
        max_completion_tokens=100,
    )
    call = transport.calls[0]
    assert call["url"] == f"{base_url}/chat/completions"
    assert call["headers"]["api-key"] == "local-test-subscription-key"
    assert "authorization" not in {key.lower() for key in call["headers"]}
    assert call["json"]["max_completion_tokens"] == 100
    assert "max_tokens" not in call["json"]
    assert result.text == "CUHK APIM test successful."
    assert result.served_model == "served-header"
    assert result.usage == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
        "reasoning_tokens": 3,
    }


def test_empty_length_response_reports_reasoning_guidance(client_factory) -> None:
    client, _ = client_factory(FakeResponse(200, chat_body("", "length")))
    result = ChatService(client).complete(
        region=Region.EUS2,
        model_id="model-router",
        system_instruction="",
        user_prompt="Test",
        max_completion_tokens=1000,
    )
    assert result.text == ""
    assert result.finish_reason == "length"
    assert "reasoning tokens reported: 3" in (result.empty_output_guidance or "")


def test_model_router_selected_model_header_takes_precedence(client_factory) -> None:
    headers = {
        "x-model-router-selected-model": "gpt-selected",
        "x-ms-served-model": "gpt-served-version",
        "x-model-router-routing-mode": "balanced",
        "x-model-router-fallback-occurred": "false",
        "apim-request-id": "apim-1",
        "x-request-id": "backend-1",
    }
    client, _ = client_factory(FakeResponse(200, chat_body(), headers))
    result = ChatService(client).complete(
        region=Region.EUS2,
        model_id="model-router",
        system_instruction="",
        user_prompt="Test",
        max_completion_tokens=1000,
    )
    assert result.served_model == "gpt-selected"
    assert result.headers.apim_request_id == "apim-1"
    assert result.headers.request_id == "backend-1"
    assert result.headers.routing["x-model-router-routing-mode"] == "balanced"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (400, {"error": {"code": "unknown_model", "message": "unknown_model"}}, "not deployed"),
        (400, {"error": {"code": "unsupported_parameter", "message": "max_tokens is unsupported"}}, "max_completion_tokens"),
        (401, {"error": {"code": "unauthorized"}}, "missing or invalid"),
        (403, {"error": {"code": "forbidden"}}, "quota"),
        (404, {"error": {"code": "not_found"}}, "operation was not found"),
    ],
)
def test_structured_http_error_guidance(client_factory, status: int, body: dict, expected: str) -> None:
    client, _ = client_factory(FakeResponse(status, body))
    with pytest.raises(APIMError, match=expected):
        client.post("chat/completions", {"model": "gpt-5.4-mini"})


def test_429_preserves_only_safe_retry_header(client_factory) -> None:
    response = FakeResponse(
        429,
        {"error": {"code": "rate_limit"}},
        {"Retry-After": "17", "api-key": "must-not-appear", "Authorization": "Bearer hidden"},
    )
    client, _ = client_factory(response)
    with pytest.raises(APIMError) as caught:
        client.post("chat/completions", {})
    assert "Retry after 17" in str(caught.value)
    assert caught.value.headers.cuhk_allowance == {"retry-after": "17"}
    assert "must-not-appear" not in repr(caught.value.headers)
    assert "hidden" not in repr(caught.value.headers)


def test_malformed_json_is_structured(client_factory) -> None:
    client, _ = client_factory(
        FakeResponse(200, raw_content=b"not-json", json_error=ValueError("bad json"))
    )
    with pytest.raises(ResponseFormatError, match="not valid JSON"):
        client.post("chat/completions", {})


def test_client_rejects_missing_key() -> None:
    with pytest.raises(APIMError, match="subscription key is missing"):
        CUHKAPIMClient("https://example.test/openai/v1", "")



def test_chat_sends_bounded_conversation_history(client_factory) -> None:
    client, transport = client_factory(FakeResponse(200, chat_body()))
    ChatService(client).complete(
        region=Region.EUS2,
        model_id="gpt-5.4-mini",
        system_instruction="Be concise.",
        user_prompt="Follow up",
        max_completion_tokens=100,
        conversation=[
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ],
    )
    assert transport.calls[0]["json"]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow up"},
    ]


def test_redirect_is_not_treated_as_success(client_factory) -> None:
    client, _ = client_factory(FakeResponse(302, chat_body()))
    with pytest.raises(APIMError) as caught:
        client.post("chat/completions", {})
    assert caught.value.status_code == 302


def test_declared_oversize_response_is_rejected_before_parsing(client_factory) -> None:
    client, _ = client_factory(
        FakeResponse(200, chat_body(), {"content-length": "99999999"})
    )
    with pytest.raises(ResponseFormatError, match="safe local response-size limit"):
        client.post("chat/completions", {})



def test_client_rejects_non_cuhk_destination() -> None:
    with pytest.raises(ValueError, match="exact documented CUHK"):
        CUHKAPIMClient("https://attacker.example/openai/v1", "test-key")


def test_client_rejects_unknown_operation(client_factory) -> None:
    client, transport = client_factory(FakeResponse())
    with pytest.raises(ValueError, match="does not allow"):
        client.post("audio/transcriptions", {})
    assert transport.calls == []


def test_operation_specific_request_size_limit(client_factory) -> None:
    client, transport = client_factory(FakeResponse())
    with pytest.raises(ResponseFormatError, match="request exceeded") as caught:
        client.post("chat/completions", {"input": "x" * 70_000})
    assert caught.value.code == "request_too_large"
    assert transport.calls == []


def test_chat_response_uses_smaller_operation_limit(client_factory) -> None:
    client, _ = client_factory(
        FakeResponse(200, chat_body(), {"content-length": str(3 * 1024 * 1024)})
    )
    with pytest.raises(ResponseFormatError, match="response exceeded"):
        client.post("chat/completions", {})


@pytest.mark.parametrize(
    ("system_instruction", "user_prompt", "expected"),
    [
        ("s" * 4_001, "test", "System instruction is limited"),
        ("", "u" * 12_001, "User prompt is limited"),
    ],
)
def test_chat_service_enforces_input_limits(
    client_factory, system_instruction: str, user_prompt: str, expected: str
) -> None:
    client, transport = client_factory(FakeResponse(200, chat_body()))
    with pytest.raises(ValueError, match=expected):
        ChatService(client).complete(
            region=Region.EUS2,
            model_id="gpt-5.4-mini",
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            max_completion_tokens=100,
        )
    assert transport.calls == []


def test_chat_service_rejects_excessive_visible_output(client_factory) -> None:
    client, _ = client_factory(FakeResponse(200, chat_body("x" * 64_001)))
    with pytest.raises(ResponseFormatError, match="chat response exceeded"):
        ChatService(client).complete(
            region=Region.EUS2,
            model_id="gpt-5.4-mini",
            system_instruction="",
            user_prompt="test",
            max_completion_tokens=100,
        )



def test_production_transport_disables_redirect_following(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StreamingResponse:
        status_code = 302

        def __init__(self) -> None:
            self.headers = {"location": "https://attacker.example/collect"}

        def iter_bytes(self):
            yield b'{"error":{"code":"redirect"}}'

    class StreamContext:
        def __enter__(self):
            return StreamingResponse()

        def __exit__(self, *args):
            return False

    class ProductionClientFake:
        def __init__(self, *, follow_redirects: bool) -> None:
            captured["follow_redirects"] = follow_redirects

        def stream(self, method: str, url: str, **kwargs):
            captured["url"] = url
            return StreamContext()

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=ProductionClientFake))
    client = CUHKAPIMClient(
        "https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1",
        "local-test-subscription-key",
    )
    with pytest.raises(APIMError) as caught:
        client.post("chat/completions", {})
    assert caught.value.status_code == 302
    assert captured == {
        "follow_redirects": False,
        "url": "https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1/chat/completions",
    }


def test_production_stream_limit_rejects_chunked_body_without_content_length(
    monkeypatch,
) -> None:
    class StreamingResponse:
        status_code = 200

        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def iter_bytes(self):
            yield b"123456"
            yield b"789012"

    class StreamContext:
        def __enter__(self):
            return StreamingResponse()

        def __exit__(self, *args):
            return False

    class ProductionClientFake:
        def __init__(self, *, follow_redirects: bool) -> None:
            assert follow_redirects is False

        def stream(self, method: str, url: str, **kwargs):
            return StreamContext()

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=ProductionClientFake))
    client = CUHKAPIMClient(
        "https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1",
        "local-test-subscription-key",
        max_response_bytes=10,
    )
    with pytest.raises(ResponseFormatError, match="response exceeded"):
        client.post("chat/completions", {})
