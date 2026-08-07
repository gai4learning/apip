from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app import require_loopback_server
from clients.cuhk_apim_client import CUHKAPIMClient
from config.settings import load_settings
from utils.errors import APIMError, sanitized_unexpected_error
from utils.security import mask_api_key, redact_text, sanitize_mapping
from utils.session import (
    MAX_CHAT_HISTORY_CHARACTERS,
    MAX_CHAT_HISTORY_MESSAGES,
    MAX_SESSION_RECORDS,
    append_bounded_record,
    bounded_chat_history,
)
from utils.usage import MAX_METADATA_VALUE_CHARS, safe_session_export, summarize_headers


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


def test_generic_azure_api_key_is_not_used(monkeypatch) -> None:
    monkeypatch.delenv("CUHK_APIM_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_API_KEY", "unrelated-azure-secret")
    assert load_settings().api_key == ""


def test_no_secret_in_transport_error_output(client_factory) -> None:
    class FailingTransport:
        def post(self, *args, **kwargs):
            raise RuntimeError(f"failed with api-key: {kwargs['headers']['api-key']}")

    key = "do-not-leak-this-key"
    client = CUHKAPIMClient(
        "https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1",
        key,
        http_client=FailingTransport(),
    )
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
    assert (3, 11) <= sys.version_info[:2] < (3, 12)


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



def test_metadata_values_and_backend_header_count_are_bounded() -> None:
    headers = {f"x-ratelimit-test-{index}": "x" * 1_000 for index in range(30)}
    headers["x-request-id"] = "safe\nvalue" + ("z" * 1_000)
    summary = summarize_headers(headers)
    assert len(summary.backend_capacity) == 16
    assert len(summary.request_id or "") == MAX_METADATA_VALUE_CHARS
    assert "\n" not in (summary.request_id or "")


def test_session_records_and_exports_are_bounded() -> None:
    records: list[dict[str, object]] = []
    for index in range(MAX_SESSION_RECORDS + 20):
        append_bounded_record(records, {"status_code": index, "served_model": "x" * 1_000})
    assert len(records) == MAX_SESSION_RECORDS
    assert len(records[0]["served_model"]) == MAX_METADATA_VALUE_CHARS
    assert set(records[0]) == {"status_code", "served_model"}
    export = safe_session_export(records)
    assert len(export["records"]) == MAX_SESSION_RECORDS
    assert len(export["records"][0]["served_model"]) == MAX_METADATA_VALUE_CHARS


def test_chat_history_is_bounded_by_count_and_characters() -> None:
    history = [
        {"role": "user", "content": str(index) * 2_000}
        for index in range(MAX_CHAT_HISTORY_MESSAGES + 5)
    ]
    bounded = bounded_chat_history(history)
    assert len(bounded) <= MAX_CHAT_HISTORY_MESSAGES
    assert sum(len(message["content"]) for message in bounded) <= MAX_CHAT_HISTORY_CHARACTERS
    assert bounded[-1]["content"].startswith(str(MAX_CHAT_HISTORY_MESSAGES + 4))


@pytest.mark.parametrize("address", ["127.0.0.1", "::1", "localhost"])
def test_loopback_server_addresses_are_allowed(address: str) -> None:
    require_loopback_server(address)


@pytest.mark.parametrize("address", [None, "0.0.0.0", "10.0.0.2", "example.test"])
def test_non_loopback_server_addresses_are_rejected(address: str | None) -> None:
    with pytest.raises(ValueError, match="loopback"):
        require_loopback_server(address)


def test_streamlit_config_binds_to_loopback() -> None:
    config = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    assert 'address = "127.0.0.1"' in config
    assert "enableXsrfProtection = true" in config
    assert "enableStaticServing = false" in config


def test_dependency_candidates_use_required_exact_versions() -> None:
    from scripts.check_dependency_lock import main

    main()
    runtime = Path("requirements.lock").read_text(encoding="utf-8").lower()
    assert "streamlit==1.61.1" in runtime
    assert "httpx==0.28.1" in runtime
    assert "python-dotenv==1.2.2" in runtime
    assert "pillow==12.3.0" in runtime
    assert ">=" not in runtime and "<" not in runtime


def test_model_output_is_rendered_through_plain_text_boundary(monkeypatch) -> None:
    import app

    rendered: list[str] = []

    class TextOnlyRenderer:
        def text(self, value: str) -> None:
            rendered.append(value)

        def write(self, value: str) -> None:
            raise AssertionError(f"Markdown-capable write used for untrusted text: {value}")

    payload = "[deceptive link](https://attacker.example) ![](https://attacker.example/pixel)"
    monkeypatch.setattr(app, "st", TextOnlyRenderer(), raising=False)
    app._render_untrusted_text(payload)
    assert rendered == [payload]
