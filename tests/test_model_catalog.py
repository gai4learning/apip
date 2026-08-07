from __future__ import annotations

from pathlib import Path

import pytest

from config.model_catalog import (
    DEFAULT_BASE_URLS,
    MODEL_CATALOG,
    Operation,
    Region,
    embedding_models,
    get_model,
    image_models,
    require_model_operation,
)


def test_catalogue_contains_required_fields() -> None:
    required = {
        "model_id", "display_name", "region", "api_name", "base_url", "capability",
        "operation", "description", "supports_chat", "supports_responses", "supports_tools",
        "supports_vision", "supports_image_generation", "supports_embeddings",
        "supports_realtime", "supports_audio", "recommended_for_starter", "validation_status",
    }
    for model in MODEL_CATALOG:
        assert required <= set(model.__dataclass_fields__)
        assert model.base_url == DEFAULT_BASE_URLS[model.region]


def test_no_duplicate_regional_model_ids_within_capability() -> None:
    keys = [(model.region, model.capability, model.model_id) for model in MODEL_CATALOG]
    assert len(keys) == len(set(keys))


def test_image_models_have_exact_regional_mapping() -> None:
    assert [model.model_id for model in image_models(Region.EUS2)] == ["gpt-image-2"]
    assert [model.model_id for model in image_models(Region.WUS3)] == ["gpt-image-1.5"]
    assert get_model(Region.EUS2, "gpt-image-2").supports_image_generation
    assert get_model(Region.WUS3, "gpt-image-1.5").supports_image_generation


@pytest.mark.parametrize(
    ("region", "invalid_model"),
    [(Region.EUS2, "gpt-image-1.5"), (Region.WUS3, "gpt-image-2")],
)
def test_invalid_model_region_pairs_are_rejected(region: Region, invalid_model: str) -> None:
    with pytest.raises(ValueError, match="not deployed"):
        get_model(region, invalid_model)


def test_embedding_models_are_eus2_only() -> None:
    expected = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}
    assert {model.model_id for model in embedding_models()} == expected
    assert all(model.region is Region.EUS2 for model in embedding_models())


def test_operation_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not use"):
        require_model_operation(Region.EUS2, "gpt-image-2", Operation.CHAT_COMPLETIONS)


def test_known_chat_validation_states() -> None:
    for region, model_id in (
        (Region.EUS2, "gpt-5.4-mini"),
        (Region.EUS2, "gpt-5.4"),
        (Region.EUS2, "model-router"),
        (Region.WUS3, "gpt-5.6-sol"),
    ):
        assert get_model(region, model_id).ui_status == "Validated"


def test_readme_lists_every_catalogue_model_and_current_endpoints() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for model in MODEL_CATALOG:
        assert f"`{model.model_id}`" in readme
    for base_url in DEFAULT_BASE_URLS.values():
        assert base_url in readme
    assert "/openai-eus2/" not in readme
    assert "gpt-4o-mini" not in readme
