from __future__ import annotations

import base64

import pytest
from conftest import FakeResponse

from config.model_catalog import Region
from services.image_service import ImageService
from utils.errors import ResponseFormatError


def test_image_prompt_limit_is_enforced_before_request(client_factory) -> None:
    client, transport = client_factory(FakeResponse())
    with pytest.raises(ValueError, match="Image prompt is limited"):
        ImageService(client).generate(
            region=Region.EUS2,
            model_id="gpt-image-2",
            prompt="x" * 4_001,
        )
    assert transport.calls == []


def test_image_count_mismatch_is_rejected_before_base64_decode(client_factory) -> None:
    client, _ = client_factory(
        FakeResponse(
            200,
            {"data": [{"b64_json": "not*base64"}, {"b64_json": "also*invalid"}]},
        )
    )
    with pytest.raises(ResponseFormatError) as caught:
        ImageService(client).generate(
            region=Region.EUS2,
            model_id="gpt-image-2",
            prompt="test",
            n=1,
        )
    assert caught.value.code == "image_count_mismatch"


def test_image_container_is_checked_before_pillow_dispatch(client_factory) -> None:
    unsupported = base64.b64encode(b"not-an-image-container").decode()
    client, _ = client_factory(
        FakeResponse(200, {"data": [{"b64_json": unsupported}]})
    )
    with pytest.raises(ResponseFormatError) as caught:
        ImageService(client).generate(
            region=Region.EUS2,
            model_id="gpt-image-2",
            prompt="test",
        )
    assert caught.value.code == "unexpected_image_container"
