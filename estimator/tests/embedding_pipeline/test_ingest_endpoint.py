"""POST /embeddings/ingest with the embedder replaced by a fake (no tokens)."""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_embedder
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.main import app
from tests.embedding_pipeline.fakes import FakeOpenAI, load_sample


@pytest.fixture
def override_embedder():
    """Install an embedder (or None) for one test and always clean up."""

    def _install(embedder):
        # NOTA: dependency_overrides cambia lo que devuelve Depends(get_embedder)
        # solo durante el test; el router no se entera de que no es OpenAI.
        app.dependency_overrides[get_embedder] = lambda: embedder

    yield _install
    app.dependency_overrides.pop(get_embedder, None)


def test_ingest_sample_returns_vectors_and_stats(override_embedder):
    fake = FakeOpenAI()
    override_embedder(OpenAIEmbedder(fake))

    response = TestClient(app).post("/embeddings/ingest", json={"budgets": load_sample()})

    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["total_budgets"] == 15
    assert body["stats"]["total_chunks"] == len(body["chunks"]) == 52
    assert body["stats"]["total_tokens"] == sum(c["token_count"] for c in body["chunks"])
    assert body["stats"]["estimated_cost_usd"] > 0
    assert body["chunks"][0]["chunk_id"] == "BUD-2024-001::AUTH-001"
    assert body["chunks"][0]["embedding"]  # cada chunk sale con su vector
    assert len(fake.embeddings.calls) == 1  # 52 chunks caben en un solo lote


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b["client_metadata"].update(sector="fintech"),  # fuera del Literal
        lambda b: b.update(components=[]),  # presupuesto sin componentes
        lambda b: b["components"][0].update(estimated_hours=-5),  # horas negativas
    ],
    ids=["unknown_sector", "no_components", "negative_hours"],
)
def test_invalid_budget_is_rejected_with_422(override_embedder, mutate):
    override_embedder(OpenAIEmbedder(FakeOpenAI()))
    budget = copy.deepcopy(load_sample()[0])
    mutate(budget)

    response = TestClient(app).post("/embeddings/ingest", json={"budgets": [budget]})

    assert response.status_code == 422


def test_missing_openai_key_returns_generic_500(override_embedder):
    override_embedder(None)

    response = TestClient(app).post("/embeddings/ingest", json={"budgets": load_sample()[:1]})

    assert response.status_code == 500
    assert response.json() == {"detail": "Embedding service is not available."}


def test_embedding_api_failure_returns_generic_500(override_embedder):
    fake = FakeOpenAI(failures=[RuntimeError("upstream exploded: secret detail")])
    override_embedder(OpenAIEmbedder(fake))

    response = TestClient(app).post("/embeddings/ingest", json={"budgets": load_sample()[:1]})

    assert response.status_code == 500
    # NOTA: el detalle interno ("secret detail") va al log, nunca al cliente.
    assert response.json() == {"detail": "Failed to generate embeddings."}


def test_endpoint_is_documented_in_openapi():
    paths = TestClient(app).get("/openapi.json").json()["paths"]
    assert "post" in paths["/embeddings/ingest"]
