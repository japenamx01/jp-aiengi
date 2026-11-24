"""OpenAIEmbedder with a fake client: batching, ordering, retries and cost."""

from __future__ import annotations

import httpx
import openai
import pytest

from app.embedding_pipeline.embedder import (
    RETRY_BACKOFF_SECONDS,
    OpenAIEmbedder,
    estimated_cost_usd,
)
from app.embedding_pipeline.schemas import Chunk
from tests.embedding_pipeline.fakes import FakeOpenAI, rate_limit_error


def make_chunks(n: int) -> list[Chunk]:
    # NOTA: textos de longitud distinta (i+1 letras) para que cada vector falso
    # sea reconocible: embedding[0] == len(text).
    return [
        Chunk(chunk_id=f"B::{i}", text="x" * (i + 1), token_count=1) for i in range(n)
    ]


def test_embed_many_batches_calls():
    client = FakeOpenAI()
    embedded = OpenAIEmbedder(client).embed_many(make_chunks(250))

    # NOTA: 250 chunks en lotes de 100 = 3 llamadas (100 + 100 + 50), no 250.
    assert [len(call) for call in client.embeddings.calls] == [100, 100, 50]
    assert len(embedded) == 250


def test_each_chunk_keeps_its_own_vector_even_if_api_reorders():
    client = FakeOpenAI(reverse=True)
    chunks = make_chunks(5)
    embedded = OpenAIEmbedder(client).embed_many(chunks)

    for chunk, item in zip(chunks, embedded):
        assert item.chunk_id == chunk.chunk_id
        assert item.embedding[0] == len(chunk.text)  # su vector, no el del vecino


def test_retries_rate_limit_with_exponential_backoff():
    waits: list[float] = []
    client = FakeOpenAI(failures=[rate_limit_error(), rate_limit_error()])

    vector = OpenAIEmbedder(client, sleep=waits.append).embed_one("hola")

    assert vector == [4.0, 0.0, 0.0]
    assert waits == [1, 2]  # falló 2 veces -> esperó 1s y 2s, y al 3er intento funcionó
    assert len(client.embeddings.calls) == 3


def test_gives_up_after_three_retries():
    waits: list[float] = []
    client = FakeOpenAI(failures=[rate_limit_error() for _ in range(4)])

    with pytest.raises(openai.RateLimitError):
        OpenAIEmbedder(client, sleep=waits.append).embed_one("hola")
    assert waits == list(RETRY_BACKOFF_SECONDS)  # 1, 2, 4 y se rinde


def test_other_errors_are_not_retried():
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    auth_error = openai.AuthenticationError(
        "bad key", response=httpx.Response(401, request=request), body=None
    )
    client = FakeOpenAI(failures=[auth_error])

    with pytest.raises(openai.AuthenticationError):
        OpenAIEmbedder(client, sleep=lambda _: pytest.fail("must not wait")).embed_one("x")
    assert len(client.embeddings.calls) == 1


def test_estimated_cost():
    # 0.02 USD por millón de tokens
    assert estimated_cost_usd(1_000_000) == pytest.approx(0.02)
    assert estimated_cost_usd(4126) == pytest.approx(0.00008252)
