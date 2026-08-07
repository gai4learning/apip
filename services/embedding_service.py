"""Embedding inspection and bounded cosine-similarity comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import fsum, hypot, isfinite
from typing import Any

from clients.cuhk_apim_client import CUHKAPIMClient
from config.model_catalog import Operation, Region, require_model_operation
from utils.errors import ResponseFormatError
from utils.usage import HeaderSummary, normalize_usage, safe_metadata_text

MAX_TEXTS = 8
MAX_TEXT_LENGTH = 4_000
MAX_TOTAL_TEXT_LENGTH = 12_000
MAX_VECTOR_DIMENSION = 4_096
MAX_TOTAL_VECTOR_SCALARS = MAX_TEXTS * MAX_VECTOR_DIMENSION


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    status_code: int
    requested_model: str
    response_model: str | None
    object_type: str | None
    vectors: tuple[tuple[float, ...], ...] = field(repr=False)
    dimension: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    headers: HeaderSummary = field(default_factory=HeaderSummary)
    latency_ms: int = 0

    @property
    def vector_count(self) -> int:
        return len(self.vectors)

    def sample(self, index: int = 0, width: int = 8) -> dict[str, tuple[float, ...]]:
        vector = self.vectors[index]
        return {"first": vector[:width], "last": vector[-width:]}


class EmbeddingService:
    def __init__(self, client: CUHKAPIMClient) -> None:
        self.client = client

    def embed(
        self,
        *,
        model_id: str,
        texts: Sequence[str],
    ) -> EmbeddingResult:
        model = require_model_operation(Region.EUS2, model_id, Operation.EMBEDDINGS)
        cleaned = [text.strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Embedding input must contain non-empty text.")
        if len(cleaned) > MAX_TEXTS:
            raise ValueError(f"Limit comparisons to at most {MAX_TEXTS} texts.")
        if any(len(text) > MAX_TEXT_LENGTH for text in cleaned):
            raise ValueError(f"Each text is limited to {MAX_TEXT_LENGTH:,} characters.")
        if sum(map(len, cleaned)) > MAX_TOTAL_TEXT_LENGTH:
            raise ValueError(f"Combined text is limited to {MAX_TOTAL_TEXT_LENGTH:,} characters.")

        payload_input: str | list[str] = cleaned[0] if len(cleaned) == 1 else cleaned
        response = self.client.post(
            Operation.EMBEDDINGS.value,
            {"model": model.model_id, "input": payload_input},
        )
        data = response.body.get("data")
        if not isinstance(data, list):
            raise _format_error(response, "missing_embedding_data", "The response did not contain embedding data.")
        if len(data) != len(cleaned):
            raise _format_error(
                response,
                "embedding_count_mismatch",
                "Embedding vector-count mismatch: similarity calculation was stopped.",
            )
        indices: list[int] = []
        for item in data:
            index = item.get("index") if isinstance(item, Mapping) else None
            if not isinstance(index, int) or isinstance(index, bool):
                raise _format_error(
                    response,
                    "invalid_embedding_index",
                    "An embedding response index was missing or invalid.",
                )
            indices.append(index)
        if sorted(indices) != list(range(len(cleaned))) or len(set(indices)) != len(indices):
            raise _format_error(
                response,
                "embedding_index_mismatch",
                "Embedding indices did not map one-to-one to the submitted texts; comparison was stopped.",
            )
        ordered = sorted(data, key=lambda item: item["index"])
        vectors: list[tuple[float, ...]] = []
        total_scalars = 0
        for item in ordered:
            raw_vector = item.get("embedding") if isinstance(item, Mapping) else None
            if not isinstance(raw_vector, list) or not raw_vector:
                raise _format_error(
                    response, "invalid_embedding", "An embedding vector was missing or empty."
                )
            if len(raw_vector) > MAX_VECTOR_DIMENSION:
                raise _format_error(
                    response,
                    "embedding_dimension_too_large",
                    "An embedding vector exceeded the safe local dimension limit.",
                )
            total_scalars += len(raw_vector)
            if total_scalars > MAX_TOTAL_VECTOR_SCALARS:
                raise _format_error(
                    response,
                    "embedding_data_too_large",
                    "The embedding response exceeded the safe local scalar limit.",
                )
            try:
                vector = tuple(_finite_float(value) for value in raw_vector)
            except (TypeError, ValueError):
                raise _format_error(
                    response,
                    "invalid_embedding",
                    "An embedding vector contained non-finite, out-of-range, or non-numeric values.",
                ) from None
            vectors.append(vector)
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise _format_error(
                response,
                "embedding_dimension_mismatch",
                "Embedding vector-dimension mismatch: similarity calculation was stopped.",
            )
        return EmbeddingResult(
            status_code=response.status_code,
            requested_model=model.model_id,
            response_model=(
                safe_metadata_text(response.body["model"])
                if isinstance(response.body.get("model"), str)
                else None
            ),
            object_type=(
                safe_metadata_text(response.body["object"])
                if isinstance(response.body.get("object"), str)
                else None
            ),
            vectors=tuple(vectors),
            dimension=dimensions.pop(),
            usage=normalize_usage(response.body),
            headers=response.headers,
            latency_ms=response.latency_ms,
        )


def _finite_float(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("Embedding values must be numeric.")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("Embedding values must fit in a finite float.") from error
    if not isfinite(converted):
        raise ValueError("Embedding values must be finite.")
    return converted


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Cosine similarity requires non-empty vectors with equal dimensions.")
    left_values = tuple(_finite_float(value) for value in left)
    right_values = tuple(_finite_float(value) for value in right)
    left_norm = hypot(*left_values)
    right_norm = hypot(*right_values)
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero-magnitude vector.")
    similarity = fsum(
        (left_value / left_norm) * (right_value / right_norm)
        for left_value, right_value in zip(left_values, right_values)
    )
    if not isfinite(similarity):
        raise ValueError("Cosine similarity could not be computed as a finite value.")
    return max(-1.0, min(1.0, similarity))


def similarity_matrix(vectors: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if len(vectors) < 2:
        raise ValueError("At least two vectors are required for comparison.")
    return tuple(
        tuple(cosine_similarity(left, right) for right in vectors)
        for left in vectors
    )


def _format_error(response: Any, code: str, guidance: str) -> ResponseFormatError:
    return ResponseFormatError(response.status_code, code, guidance, response.headers)
