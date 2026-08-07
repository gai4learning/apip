"""Chat Completions requests and safe response parsing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from clients.cuhk_apim_client import APIMResponse, CUHKAPIMClient
from config.model_catalog import Operation, Region, require_model_operation
from utils.errors import ResponseFormatError
from utils.usage import HeaderSummary, normalize_usage


@dataclass(frozen=True, slots=True)
class ChatResult:
    status_code: int
    requested_model: str
    served_model: str | None
    region: Region
    text: str
    finish_reason: str | None
    usage: dict[str, int]
    headers: HeaderSummary
    latency_ms: int
    empty_output_guidance: str | None = None


class ChatService:
    def __init__(self, client: CUHKAPIMClient) -> None:
        self.client = client

    def complete(
        self,
        *,
        region: Region,
        model_id: str,
        system_instruction: str,
        user_prompt: str,
        max_completion_tokens: int,
        temperature: float | None = None,
        conversation: Sequence[Mapping[str, str]] = (),
    ) -> ChatResult:
        model = require_model_operation(region, model_id, Operation.CHAT_COMPLETIONS)
        if not user_prompt.strip():
            raise ValueError("Enter a user prompt before submitting.")
        if not 1 <= max_completion_tokens <= 16_384:
            raise ValueError("Maximum completion tokens must be between 1 and 16,384.")
        messages: list[dict[str, str]] = []
        if system_instruction.strip():
            messages.append({"role": "system", "content": system_instruction.strip()})
        history = _validated_conversation(conversation)
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt.strip()})
        payload: dict[str, Any] = {
            "model": model.model_id,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        if temperature is not None:
            if not model.supports_temperature:
                raise ValueError(f"Temperature is not enabled for {model.model_id} in this catalogue.")
            payload["temperature"] = temperature

        response = self.client.post(Operation.CHAT_COMPLETIONS.value, payload)
        return _parse_chat_response(response, region, model.model_id)


def _validated_conversation(
    conversation: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if len(conversation) > 20:
        conversation = conversation[-20:]
    validated: list[dict[str, str]] = []
    total_characters = 0
    for message in conversation:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError("Conversation history contains an unsupported message.")
        total_characters += len(content)
        if total_characters > 24_000:
            raise ValueError("Conversation history exceeds the 24,000-character local limit.")
        validated.append({"role": role, "content": content})
    return validated


def _parse_chat_response(
    response: APIMResponse, region: Region, requested_model: str
) -> ChatResult:
    body = response.body
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ResponseFormatError(
            response.status_code,
            "missing_choices",
            "The Chat Completions response did not contain a usable choice.",
            response.headers,
        )
    choice = choices[0]
    message = choice.get("message")
    content: Any = message.get("content") if isinstance(message, Mapping) else ""
    if isinstance(content, list):
        text = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("type") in {"text", "output_text"}
        )
    else:
        text = content if isinstance(content, str) else ""
    finish_reason = choice.get("finish_reason")
    finish_reason = finish_reason if isinstance(finish_reason, str) else None
    usage = normalize_usage(body)
    empty_guidance = None
    if not text.strip():
        if finish_reason == "length":
            reasoning = usage.get("reasoning_tokens", 0)
            empty_guidance = (
                "No visible answer was returned because the completion allowance was exhausted"
                f" (reasoning tokens reported: {reasoning}). Increase max_completion_tokens."
            )
        else:
            empty_guidance = "The response was successful but contained no visible assistant text."
    served_model = (
        response.headers.routing.get("x-model-router-selected-model")
        or response.headers.routing.get("x-ms-served-model")
        or (body.get("model") if isinstance(body.get("model"), str) else None)
    )
    return ChatResult(
        status_code=response.status_code,
        requested_model=requested_model,
        served_model=served_model,
        region=region,
        text=text,
        finish_reason=finish_reason,
        usage=usage,
        headers=response.headers,
        latency_ms=response.latency_ms,
        empty_output_guidance=empty_guidance,
    )
