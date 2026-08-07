from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from clients.cuhk_apim_client import CUHKAPIMClient
from config.settings import load_settings
from utils.errors import APIMError, sanitized_unexpected_error
from utils.security import mask_api_key, redact_text, sanitize_mapping
from utils.usage import safe_session_export, summarize_headers


def test_api_key_and_authorization_redaction() -> None:
    secret = "subscription-secret-1234"
    text = f"api-key: {secret}\nAuthorization: Bearer bearer-secret\nother=value"
    redacted = redact_text(text, (secret,))
    assert secret not in redacted
    assert "bearer-secret" not in redacted
    assert redacted.count("[REDACTED]") >= 2
    assert mask_api_key(secret) == "Configured"


def test_sensitive_headers_are_not_summarized() -> None:
    summary = summarize_headers(
        {
            "api-key": "hidden-key",
            "Authorization": "Bearer hidden-token",
            "x-cuhk-tpm-remaining": "200000",
            "x-ratelimit-remaining-requests": "99",
            "x-request-id": "safe-request-id",
        }
    )
    rendered = repr(summary)
    assert "hidden-key" not in rendered
    assert "hidden-token" not in rendered
    assert summary.cuhk_allowance == {"x-cuhk-tpm-remaining": "200000"}
    assert summary.backend_capacity == {"x-ratelimit-remaining-requests": "99"}


def test_environment_loads_without_key(monkeypatch) -> None:
    for name in (
        "CUHK_APIM_API_KEY", "AZURE_API_KEY", "CUHK_EUS2_BASE_URL", "CUHK_WUS3_BASE_URL",
        "CUHK_DEFAULT_REGION", "CUHK_DEFAULT_CHAT_MODEL", "CUHK_DEFAULT_IMAGE_MODEL",
        "CUHK_DEFAULT_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings()
    assert settings.api_key == ""
    with pytest.raises(APIMError):
        CUHKAPIMClient(settings.eus2_base_url, settings.api_key)


def test_azure_api_key_migration_alias(monkeypatch) -> None:
    monkeypatch.delenv("CUHK_APIM_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_API_KEY", "kiro-web-secret-value")
    assert load_settings().api_key == "kiro-web-secret-value"


def test_no_secret_in_transport_error_output(client_factory) -> None:
    class FailingTransport:
        def post(self, *args, **kwargs):
            raise RuntimeError(f"failed with api-key: {kwargs['headers']['api-key']}")

    key = "do-not-leak-this-key"
    client = CUHKAPIMClient("https://example.test/openai/v1", key, http_client=FailingTransport())
    with pytest.raises(APIMError) as caught:
        client.post("chat/completions", {})
    assert key not in str(caught.value)
    assert key not in repr(caught.value)
    assert key not in sanitized_unexpected_error(RuntimeError(key), key)


def test_sanitized_mapping_and_export_exclude_sensitive_payloads() -> None:
    source = {
        "api-key": "secret",
        "prompt": "private prompt",
        "b64_json": "base64data",
        "embedding": [1.0, 2.0],
        "status_code": 200,
    }
    sanitized = sanitize_mapping(source)
    export = safe_session_export([source])
    rendered = json.dumps({"mapping": sanitized, "export": export})
    for forbidden in ("secret", "private prompt", "base64data", "1.0"):
        assert forbidden not in rendered
    assert "200" in rendered


def test_env_example_contains_no_key_and_current_defaults() -> None:
    example = Path(".env.example").read_text(encoding="utf-8")
    assert "CUHK_APIM_API_KEY=\n" in example
    assert "foundry-eus2/openai/v1" in example
    assert "foundry-wus3/openai/v1" in example
    assert "gpt-5.4-mini" in example
    assert "gpt-image-2" in example
    assert "text-embedding-3-small" in example


def test_python_runtime_is_supported() -> None:
    assert sys.version_info >= (3, 11)


def test_application_imports_successfully() -> None:
    module = importlib.import_module("app")
    assert callable(module.main)


@pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None,
    reason="Streamlit is not installed in this sandbox",
)
def test_streamlit_app_starts_without_live_key(monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("CUHK_APIM_API_KEY", raising=False)
    app = AppTest.from_file("app.py", default_timeout=15).run()
    assert not app.exception
    assert any("LOCAL-ONLY" in element.value for element in app.error)
    assert any("Get Started" in element.value for element in app.title)



def test_wus3_region_derives_regional_defaults(monkeypatch) -> None:
    monkeypatch.setenv("CUHK_DEFAULT_REGION", "WUS3")
    monkeypatch.delenv("CUHK_DEFAULT_CHAT_MODEL", raising=False)
    monkeypatch.delenv("CUHK_DEFAULT_IMAGE_MODEL", raising=False)
    settings = load_settings()
    assert settings.default_chat_model == "gpt-5.6-sol"
    assert settings.default_image_model == "gpt-image-1.5"
    assert settings.default_image_region.value == "WUS3"


def test_environment_rejects_non_cuhk_api_host(monkeypatch) -> None:
    monkeypatch.setenv("CUHK_EUS2_BASE_URL", "https://attacker.example/openai/v1")
    with pytest.raises(ValueError, match="CUHK APIM HTTPS host"):
        load_settings()


def test_environment_rejects_wrong_regional_path(monkeypatch) -> None:
    monkeypatch.setenv(
        "CUHK_EUS2_BASE_URL",
        "https://cuhk-apip.azure-api.net/foundry-wus3/openai/v1",
    )
    with pytest.raises(ValueError, match="documented EUS2"):
        load_settings()
