from __future__ import annotations

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
