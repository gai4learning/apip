"""Embedding inspection and bounded cosine-similarity comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import Any

from clients.cuhk_apim_client import CUHKAPIMClient
from config.model_catalog import Operation, Region, require_model_operation
from utils.errors import ResponseFormatError
from utils.usage import HeaderSummary, normalize_usage

MAX_TEXTS = 8
MAX_TEXT_LENGTH = 4_000
MAX_TOTAL_TEXT_LENGTH = 12_000


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
        for item in ordered:
            raw_vector = item.get("embedding") if isinstance(item, Mapping) else None
            if not isinstance(raw_vector, list) or not raw_vector:
                raise _format_error(response, "invalid_embedding", "An embedding vector was missing or empty.")
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(value)
                for value in raw_vector
            ):
                raise _format_error(
                    response,
                    "invalid_embedding",
                    "An embedding vector contained non-finite or non-numeric values.",
                )
            vectors.append(tuple(float(value) for value in raw_vector))
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
            response_model=response.body.get("model") if isinstance(response.body.get("model"), str) else None,
            object_type=response.body.get("object") if isinstance(response.body.get("object"), str) else None,
            vectors=tuple(vectors),
            dimension=dimensions.pop(),
            usage=normalize_usage(response.body),
            headers=response.headers,
            latency_ms=response.latency_ms,
        )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Cosine similarity requires non-empty vectors with equal dimensions.")
    denominator = sqrt(sum(value * value for value in left)) * sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero-magnitude vector.")
    return sum(a * b for a, b in zip(left, right)) / denominator


def similarity_matrix(vectors: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if len(vectors) < 2:
        raise ValueError("At least two vectors are required for comparison.")
    return tuple(
        tuple(cosine_similarity(left, right) for right in vectors)
        for left in vectors
    )


def _format_error(response: Any, code: str, guidance: str) -> ResponseFormatError:
    return ResponseFormatError(response.status_code, code, guidance, response.headers)
