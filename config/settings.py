"""Environment configuration for the local-only CUHK APIM demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from config.model_catalog import (
    DEFAULT_BASE_URLS,
    DEFAULT_CHAT_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_IMAGE_MODELS,
    Operation,
    Region,
    require_model_operation,
)


@dataclass(frozen=True, slots=True)
class Settings:
    # This is an APIM subscription key, never a Foundry backend key.
    api_key: str
    eus2_base_url: str
    wus3_base_url: str
    default_region: Region
    default_chat_model: str
    default_image_model: str
    default_embedding_model: str

    def base_url(self, region: Region) -> str:
        return self.eus2_base_url if region is Region.EUS2 else self.wus3_base_url

    @property
    def default_image_region(self) -> Region:
        for region, model_id in DEFAULT_IMAGE_MODELS.items():
            if model_id == self.default_image_model:
                return region
        raise ValueError("The configured default image model has no regional mapping.")


def _base_url(name: str, default: str, region: Region) -> str:
    value = os.getenv(name, default).strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname != "cuhk-apip.azure-api.net":
        raise ValueError(f"{name} must use the CUHK APIM HTTPS host.")
    if parsed.port is not None or parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{name} must not contain credentials or a custom port.")
    expected_path = f"/foundry-{region.value.lower()}/openai/v1"
    if parsed.path.rstrip("/") != expected_path:
        raise ValueError(f"{name} must use the documented {region.value} OpenAI v1 path.")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not contain a query string or fragment.")
    return value


def load_settings() -> Settings:
    region_value = os.getenv("CUHK_DEFAULT_REGION", Region.EUS2.value).strip().upper()
    try:
        region = Region(region_value)
    except ValueError as error:
        raise ValueError("CUHK_DEFAULT_REGION must be EUS2 or WUS3.") from error

    settings = Settings(
        api_key=os.getenv("CUHK_APIM_API_KEY", "").strip(),
        eus2_base_url=_base_url(
            "CUHK_EUS2_BASE_URL", DEFAULT_BASE_URLS[Region.EUS2], Region.EUS2
        ),
        wus3_base_url=_base_url(
            "CUHK_WUS3_BASE_URL", DEFAULT_BASE_URLS[Region.WUS3], Region.WUS3
        ),
        default_region=region,
        default_chat_model=os.getenv(
            "CUHK_DEFAULT_CHAT_MODEL", DEFAULT_CHAT_MODELS[region]
        ).strip(),
        default_image_model=os.getenv(
            "CUHK_DEFAULT_IMAGE_MODEL", DEFAULT_IMAGE_MODELS[region]
        ).strip(),
        default_embedding_model=os.getenv(
            "CUHK_DEFAULT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ).strip(),
    )
    # Validate defaults when they belong to the configured default region. Image and
    # embedding defaults are allowed to target their documented starter regions.
    require_model_operation(
        settings.default_region, settings.default_chat_model, Operation.CHAT_COMPLETIONS
    )
    require_model_operation(
        Region.EUS2, settings.default_embedding_model, Operation.EMBEDDINGS
    )
    image_region = next(
        (
            region for region, model_id in DEFAULT_IMAGE_MODELS.items()
            if model_id == settings.default_image_model
        ),
        None,
    )
    if image_region is None:
        raise ValueError("CUHK_DEFAULT_IMAGE_MODEL must be a documented regional image model.")
    require_model_operation(
        image_region, settings.default_image_model, Operation.IMAGE_GENERATION
    )
    return settings
