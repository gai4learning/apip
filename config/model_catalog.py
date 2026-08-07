"""Single source of truth for CUHK Foundry regional model deployments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Region(StrEnum):
    EUS2 = "EUS2"
    WUS3 = "WUS3"


class Capability(StrEnum):
    CHAT = "Chat"
    ROUTING = "Routing"
    CODING = "Coding"
    IMAGE = "Image generation"
    EMBEDDING = "Embeddings"
    REALTIME = "Realtime"
    TRANSLATION = "Realtime translation"
    TRANSCRIPTION = "Transcription"


class Operation(StrEnum):
    CHAT_COMPLETIONS = "chat/completions"
    RESPONSES = "responses"
    IMAGE_GENERATION = "images/generations"
    EMBEDDINGS = "embeddings"
    REALTIME = "realtime (operation-specific)"
    AUDIO_TRANSCRIPTIONS = "audio/transcriptions (multipart)"


class ValidationStatus(StrEnum):
    VALIDATED = "Validated"
    PENDING = "Validation pending"


REGION_NAMES = {
    Region.EUS2: "East US 2",
    Region.WUS3: "West US 3",
}

DEFAULT_BASE_URLS = {
    Region.EUS2: "https://cuhk-apip.azure-api.net/foundry-eus2/openai/v1",
    Region.WUS3: "https://cuhk-apip.azure-api.net/foundry-wus3/openai/v1",
}
DEFAULT_CHAT_MODELS = {
    Region.EUS2: "gpt-5.4-mini",
    Region.WUS3: "gpt-5.6-sol",
}
DEFAULT_IMAGE_MODELS = {
    Region.EUS2: "gpt-image-2",
    Region.WUS3: "gpt-image-1.5",
}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MODEL_ROUTER_MODEL_ID = "model-router"

IMPLEMENTED_OPERATIONS = {
    Operation.CHAT_COMPLETIONS,
    Operation.IMAGE_GENERATION,
    Operation.EMBEDDINGS,
}


@dataclass(frozen=True, slots=True)
class ModelDeployment:
    model_id: str
    display_name: str
    region: Region
    api_name: str
    base_url: str
    capability: Capability
    operation: Operation
    description: str
    supports_chat: bool = False
    supports_responses: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    supports_image_generation: bool = False
    supports_embeddings: bool = False
    supports_realtime: bool = False
    supports_audio: bool = False
    recommended_for_starter: bool = False
    validation_status: ValidationStatus = ValidationStatus.PENDING
    supports_temperature: bool = False
    legacy_compatibility: bool = False

    @property
    def ui_status(self) -> str:
        if self.operation not in IMPLEMENTED_OPERATIONS:
            return "Not yet implemented"
        if self.validation_status is ValidationStatus.VALIDATED:
            return "Validated"
        return "Available for testing"


def _deployment(
    model_id: str,
    display_name: str,
    region: Region,
    capability: Capability,
    operation: Operation,
    description: str,
    **features: bool | ValidationStatus,
) -> ModelDeployment:
    return ModelDeployment(
        model_id=model_id,
        display_name=display_name,
        region=region,
        api_name=f"CUHK Foundry Models {region.value}",
        base_url=DEFAULT_BASE_URLS[region],
        capability=capability,
        operation=operation,
        description=description,
        **features,
    )


MODEL_CATALOG: tuple[ModelDeployment, ...] = (
    _deployment(
        "gpt-5.4-nano", "GPT-5.4 Nano", Region.EUS2, Capability.CHAT,
        Operation.CHAT_COMPLETIONS,
        "Efficient language deployment. Chat Completions is available for testing; supported Responses workflows require separate validation.",
        supports_chat=True, supports_responses=True,
    ),
    _deployment(
        "gpt-5.4-mini", "GPT-5.4 Mini", Region.EUS2, Capability.CHAT,
        Operation.CHAT_COMPLETIONS,
        "Recommended first general chat deployment. Chat Completions was externally validated through CUHK APIM.",
        supports_chat=True, supports_responses=True, supports_tools=True,
        supports_vision=True, recommended_for_starter=True,
        validation_status=ValidationStatus.VALIDATED,
    ),
    _deployment(
        "gpt-5.4", "GPT-5.4", Region.EUS2, Capability.CHAT,
        Operation.CHAT_COMPLETIONS,
        "Higher-capability language deployment. Chat Completions was externally validated through CUHK APIM.",
        supports_chat=True, validation_status=ValidationStatus.VALIDATED,
    ),
    _deployment(
        "gpt-5.4-pro", "GPT-5.4 Pro", Region.EUS2, Capability.CHAT,
        Operation.CHAT_COMPLETIONS,
        "Advanced language deployment; the Chat Completions combination remains pending validation.",
        supports_chat=True,
    ),
    _deployment(
        "gpt-5.3-codex", "GPT-5.3 Codex", Region.EUS2, Capability.CODING,
        Operation.RESPONSES,
        "Coding deployment intended for a supported Responses workflow. This Streamlit UI does not yet implement that operation.",
        supports_responses=True,
    ),
    _deployment(
        "model-router", "Model Router", Region.EUS2, Capability.ROUTING,
        Operation.CHAT_COMPLETIONS,
        "Routes a Chat Completions request to an eligible underlying model. The selected model may be reported in response headers.",
        supports_chat=True, recommended_for_starter=True,
        validation_status=ValidationStatus.VALIDATED,
    ),
    _deployment(
        "gpt-image-2", "GPT Image 2", Region.EUS2, Capability.IMAGE,
        Operation.IMAGE_GENERATION,
        "East US 2 image-generation deployment. The regional mapping is known; live operation validation remains pending.",
        supports_image_generation=True, recommended_for_starter=True,
    ),
    _deployment(
        "gpt-realtime-2", "GPT Realtime 2", Region.EUS2, Capability.REALTIME,
        Operation.REALTIME,
        "Realtime deployment requiring an operation-specific session protocol not implemented by this synchronous UI.",
        supports_realtime=True, supports_audio=True,
    ),
    _deployment(
        "gpt-realtime-2.1", "GPT Realtime 2.1", Region.EUS2, Capability.REALTIME,
        Operation.REALTIME,
        "Realtime deployment requiring an operation-specific session protocol not implemented by this synchronous UI.",
        supports_realtime=True, supports_audio=True,
    ),
    _deployment(
        "gpt-realtime-2.1-mini", "GPT Realtime 2.1 Mini", Region.EUS2, Capability.REALTIME,
        Operation.REALTIME,
        "Realtime deployment requiring an operation-specific session protocol not implemented by this synchronous UI.",
        supports_realtime=True, supports_audio=True,
    ),
    _deployment(
        "gpt-realtime-translate", "GPT Realtime Translate", Region.EUS2, Capability.TRANSLATION,
        Operation.REALTIME,
        "Realtime translation deployment; the required operation-specific protocol is not implemented here.",
        supports_realtime=True, supports_audio=True,
    ),
    _deployment(
        "gpt-realtime-whisper", "GPT Realtime Whisper", Region.EUS2, Capability.TRANSCRIPTION,
        Operation.REALTIME,
        "Realtime speech-recognition deployment; it must not be routed through Chat Completions.",
        supports_realtime=True, supports_audio=True,
    ),
    _deployment(
        "gpt-4o-transcribe", "GPT-4o Transcribe", Region.EUS2, Capability.TRANSCRIPTION,
        Operation.AUDIO_TRANSCRIPTIONS,
        "Audio transcription deployment requiring a multipart transcription operation not implemented by this UI.",
        supports_audio=True,
    ),
    _deployment(
        "text-embedding-3-small", "Text Embedding 3 Small", Region.EUS2, Capability.EMBEDDING,
        Operation.EMBEDDINGS,
        "Recommended first embedding deployment for new demonstrations; live operation validation remains pending.",
        supports_embeddings=True, recommended_for_starter=True,
    ),
    _deployment(
        "text-embedding-3-large", "Text Embedding 3 Large", Region.EUS2, Capability.EMBEDDING,
        Operation.EMBEDDINGS,
        "Embedding deployment available for comparative evaluation; live operation validation remains pending.",
        supports_embeddings=True,
    ),
    _deployment(
        "text-embedding-ada-002", "Text Embedding Ada 002 (Legacy)", Region.EUS2, Capability.EMBEDDING,
        Operation.EMBEDDINGS,
        "Legacy compatibility deployment. Evaluate text-embedding-3-small first for new demonstrations.",
        supports_embeddings=True, legacy_compatibility=True,
    ),
    _deployment(
        "gpt-5.6-sol", "GPT-5.6 Sol", Region.WUS3, Capability.CHAT,
        Operation.CHAT_COMPLETIONS,
        "West US 3 chat deployment externally validated through CUHK APIM.",
        supports_chat=True, recommended_for_starter=True,
        validation_status=ValidationStatus.VALIDATED,
    ),
    _deployment(
        "gpt-5.6-luna", "GPT-5.6 Luna", Region.WUS3, Capability.CHAT,
        Operation.CHAT_COMPLETIONS,
        "West US 3 chat deployment available for testing; service-owner validation remains pending.",
        supports_chat=True,
    ),
    _deployment(
        "gpt-5.6-terra", "GPT-5.6 Terra", Region.WUS3, Capability.CHAT,
        Operation.CHAT_COMPLETIONS,
        "West US 3 chat deployment available for testing; service-owner validation remains pending.",
        supports_chat=True,
    ),
    _deployment(
        "gpt-image-1.5", "GPT Image 1.5", Region.WUS3, Capability.IMAGE,
        Operation.IMAGE_GENERATION,
        "West US 3 image-generation deployment. The regional mapping is known; live operation validation remains pending.",
        supports_image_generation=True, recommended_for_starter=True,
    ),
)


def models_for_region(region: Region) -> tuple[ModelDeployment, ...]:
    return tuple(model for model in MODEL_CATALOG if model.region is region)


def models_for_operation(region: Region, operation: Operation) -> tuple[ModelDeployment, ...]:
    return tuple(
        model for model in MODEL_CATALOG
        if model.region is region and model.operation is operation
    )


def chat_models(region: Region) -> tuple[ModelDeployment, ...]:
    return tuple(model for model in models_for_region(region) if model.supports_chat)


def image_models(region: Region) -> tuple[ModelDeployment, ...]:
    return tuple(model for model in models_for_region(region) if model.supports_image_generation)


def embedding_models() -> tuple[ModelDeployment, ...]:
    return tuple(model for model in models_for_region(Region.EUS2) if model.supports_embeddings)


def get_model(region: Region, model_id: str) -> ModelDeployment:
    for model in MODEL_CATALOG:
        if model.region is region and model.model_id == model_id:
            return model
    raise ValueError(f"Model '{model_id}' is not deployed in {REGION_NAMES[region]}.")


def require_model_operation(
    region: Region, model_id: str, operation: Operation
) -> ModelDeployment:
    model = get_model(region, model_id)
    if model.operation is not operation:
        raise ValueError(
            f"Model '{model_id}' does not use the {operation.value} operation in {REGION_NAMES[region]}."
        )
    return model
