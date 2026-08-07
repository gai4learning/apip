from __future__ import annotations

import base64
import io
import logging

import pytest
from conftest import FakeResponse

pytest.importorskip("PIL")
from PIL import Image

from config.model_catalog import Region
from services.image_service import ImageService
from utils.errors import ResponseFormatError
from utils.usage import safe_session_export


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (1024, 1024), color=(20, 40, 60)).save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize(
    ("region", "model_id", "base_url"),
    [
        (Region.EUS2, "gpt-image-2", "https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1"),
        (Region.WUS3, "gpt-image-1.5", "https://cuhk-apip.azure-api.net/foundry-wus3/openai/v1"),
    ],
)
def test_regional_image_url_default_n_and_base64_decode(
    client_factory, region: Region, model_id: str, base_url: str
) -> None:
    encoded = base64.b64encode(png_bytes()).decode()
    client, transport = client_factory(
        FakeResponse(200, {"data": [{"b64_json": encoded}]}), base_url
    )
    result = ImageService(client).generate(
        region=region,
        model_id=model_id,
        prompt="Abstract university platform",
    )
    call = transport.calls[0]
    assert call["url"] == f"{base_url}/images/generations"
    assert call["json"]["n"] == 1
    assert call["json"]["quality"] == "low"
    assert result.images[0].content == png_bytes()
    assert result.images[0].media_type == "image/png"


def test_invalid_regional_image_pair_is_rejected_before_request(client_factory) -> None:
    client, transport = client_factory(FakeResponse())
    with pytest.raises(ValueError, match="not deployed"):
        ImageService(client).generate(
            region=Region.EUS2,
            model_id="gpt-image-1.5",
            prompt="test",
        )
    assert transport.calls == []


def test_starter_mode_rejects_multiple_images(client_factory) -> None:
    client, _ = client_factory(FakeResponse())
    with pytest.raises(ValueError, match="n = 1"):
        ImageService(client).generate(
            region=Region.EUS2,
            model_id="gpt-image-2",
            prompt="test",
            n=2,
            starter_mode=True,
        )


def test_malformed_base64_is_rejected(client_factory) -> None:
    client, _ = client_factory(FakeResponse(200, {"data": [{"b64_json": "not*base64"}]}))
    with pytest.raises(ResponseFormatError, match="malformed Base64"):
        ImageService(client).generate(
            region=Region.EUS2, model_id="gpt-image-2", prompt="test"
        )


@pytest.mark.parametrize("body", [{}, {"data": []}, {"data": [{}]}])
def test_missing_image_data_is_cleanly_rejected(client_factory, body: dict) -> None:
    client, _ = client_factory(FakeResponse(200, body, {"apim-request-id": "safe-id"}))
    with pytest.raises(ResponseFormatError) as caught:
        ImageService(client).generate(
            region=Region.EUS2, model_id="gpt-image-2", prompt="test"
        )
    assert caught.value.headers.apim_request_id == "safe-id"
    assert caught.value.code in {"missing_image_data", "missing_b64_json"}


def test_base64_is_absent_from_logs_repr_and_exports(client_factory, caplog) -> None:
    encoded = base64.b64encode(png_bytes()).decode()
    logger = logging.getLogger("image-test")
    client, _ = client_factory(FakeResponse(200, {"data": [{"b64_json": encoded}]}))
    client._logger = logger
    with caplog.at_level(logging.INFO, logger="image-test"):
        result = ImageService(client).generate(
            region=Region.EUS2, model_id="gpt-image-2", prompt="private prompt"
        )
    export = safe_session_export(
        [{"operation": "images/generations", "b64_json": encoded, "prompt": "private prompt", "count": 1}]
    )
    assert encoded not in caplog.text
    assert encoded not in repr(result)
    assert encoded not in repr(export)
    assert "private prompt" not in repr(export)



def test_truncated_png_is_rejected(client_factory) -> None:
    truncated = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x04\x00"
        b"\x00\x00\x04\x00"
    )
    encoded = base64.b64encode(truncated).decode()
    client, _ = client_factory(FakeResponse(200, {"data": [{"b64_json": encoded}]}))
    with pytest.raises(ResponseFormatError, match="complete image"):
        ImageService(client).generate(
            region=Region.EUS2, model_id="gpt-image-2", prompt="test"
        )
