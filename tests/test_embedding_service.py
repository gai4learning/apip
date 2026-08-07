from __future__ import annotations

import math

import pytest
from conftest import FakeResponse

from services.embedding_service import (
    EmbeddingService,
    cosine_similarity,
    similarity_matrix,
)
from utils.errors import ResponseFormatError
from utils.usage import safe_session_export


def embedding_body(vectors: list[list[float]]) -> dict:
    return {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 9, "total_tokens": 9},
    }


def test_single_embedding_metadata_and_sample(client_factory) -> None:
    client, transport = client_factory(FakeResponse(200, embedding_body([[1.0, 2.0, 3.0]])))
    result = EmbeddingService(client).embed(
        model_id="text-embedding-3-small", texts=["CUHK embedding test"]
    )
    assert transport.calls[0]["url"].endswith("/embeddings")
    assert transport.calls[0]["json"]["input"] == "CUHK embedding test"
    assert result.vector_count == 1
    assert result.dimension == 3
    assert result.object_type == "list"
    assert result.usage == {"prompt_tokens": 9, "total_tokens": 9}
    assert result.sample(width=2) == {"first": (1.0, 2.0), "last": (2.0, 3.0)}


def test_multiple_texts_use_one_request_and_preserve_vector_count(client_factory) -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    client, transport = client_factory(FakeResponse(200, embedding_body(vectors)))
    texts = ["one", "two", "three"]
    result = EmbeddingService(client).embed(model_id="text-embedding-3-small", texts=texts)
    assert len(transport.calls) == 1
    assert transport.calls[0]["json"]["input"] == texts
    assert result.vector_count == len(texts)


def test_vector_count_mismatch_stops_comparison(client_factory) -> None:
    client, _ = client_factory(FakeResponse(200, embedding_body([[1.0, 0.0]])))
    with pytest.raises(ResponseFormatError, match="vector-count mismatch"):
        EmbeddingService(client).embed(
            model_id="text-embedding-3-small", texts=["one", "two"]
        )


def test_vector_dimension_mismatch_stops_comparison(client_factory) -> None:
    client, _ = client_factory(
        FakeResponse(200, embedding_body([[1.0, 0.0], [1.0, 0.0, 2.0]]))
    )
    with pytest.raises(ResponseFormatError, match="vector-dimension mismatch"):
        EmbeddingService(client).embed(
            model_id="text-embedding-3-small", texts=["one", "two"]
        )


def test_cosine_similarity_and_matrix() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    matrix = similarity_matrix([[1, 0], [1, 1]])
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[1][1] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(1 / math.sqrt(2))


def test_cosine_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero-magnitude"):
        cosine_similarity([0, 0], [1, 0])


def test_full_vectors_are_not_in_repr_or_export(client_factory) -> None:
    distinctive = [0.123456789, 0.987654321, 0.444444444]
    client, _ = client_factory(FakeResponse(200, embedding_body([distinctive])))
    result = EmbeddingService(client).embed(
        model_id="text-embedding-3-small", texts=["sensitive source text"]
    )
    export = safe_session_export(
        [{"vector": distinctive, "input": "sensitive source text", "vector_count": 1, "vector_dimension": 3}]
    )
    assert "0.123456789" not in repr(result)
    assert "0.123456789" not in repr(export)
    assert "sensitive source text" not in repr(export)



def test_duplicate_embedding_indices_are_rejected(client_factory) -> None:
    body = embedding_body([[1.0, 0.0], [0.0, 1.0]])
    body["data"][1]["index"] = 0
    client, _ = client_factory(FakeResponse(200, body))
    with pytest.raises(ResponseFormatError, match="one-to-one"):
        EmbeddingService(client).embed(
            model_id="text-embedding-3-small", texts=["one", "two"]
        )


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_embedding_values_are_rejected(client_factory, non_finite: float) -> None:
    client, _ = client_factory(FakeResponse(200, embedding_body([[1.0, non_finite]])))
    with pytest.raises(ResponseFormatError, match="non-finite"):
        EmbeddingService(client).embed(
            model_id="text-embedding-3-small", texts=["one"]
        )



def test_embedding_dimension_limit_is_enforced_before_copying(client_factory) -> None:
    oversized = [0.0] * 4_097
    client, _ = client_factory(FakeResponse(200, embedding_body([oversized])))
    with pytest.raises(ResponseFormatError, match="dimension limit"):
        EmbeddingService(client).embed(
            model_id="text-embedding-3-small", texts=["one"]
        )



def test_huge_integer_embedding_value_is_rejected(client_factory) -> None:
    client, _ = client_factory(FakeResponse(200, embedding_body([[1.0, 10**1000]])))
    with pytest.raises(ResponseFormatError, match="out-of-range"):
        EmbeddingService(client).embed(
            model_id="text-embedding-3-small", texts=["one"]
        )


def test_extreme_finite_vectors_produce_finite_similarity() -> None:
    similarity = cosine_similarity([1e308, 1e308], [1e308, -1e308])
    assert math.isfinite(similarity)
    assert similarity == pytest.approx(0.0)


def test_embedding_metadata_is_bounded(client_factory) -> None:
    body = embedding_body([[1.0, 0.0]])
    body["model"] = "m" * 1_000
    body["object"] = "o" * 1_000
    client, _ = client_factory(FakeResponse(200, body))
    result = EmbeddingService(client).embed(
        model_id="text-embedding-3-small", texts=["one"]
    )
    assert len(result.response_model or "") == 256
    assert len(result.object_type or "") == 256


def test_aggregate_embedding_scalar_limit_is_enforced(client_factory, monkeypatch) -> None:
    import services.embedding_service as embedding_module

    monkeypatch.setattr(embedding_module, "MAX_TOTAL_VECTOR_SCALARS", 3)
    client, _ = client_factory(
        FakeResponse(200, embedding_body([[1.0, 0.0], [0.0, 1.0]]))
    )
    with pytest.raises(ResponseFormatError, match="scalar limit"):
        EmbeddingService(client).embed(
            model_id="text-embedding-3-small", texts=["one", "two"]
        )
