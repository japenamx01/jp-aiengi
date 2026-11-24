"""Structural chunker: pure function, so the tests need no network or mocks."""

from __future__ import annotations

import pytest

from app.embedding_pipeline.chunker import JSONStructuralChunker, count_tokens
from app.embedding_pipeline.schemas import Budget
from tests.embedding_pipeline.fakes import load_sample


@pytest.fixture(scope="module")
def budgets() -> list[Budget]:
    return [Budget.model_validate(item) for item in load_sample()]


@pytest.fixture(scope="module")
def chunks(budgets):
    return JSONStructuralChunker().chunk(budgets)


def test_one_chunk_per_component(budgets, chunks):
    # NOTA: la regla central del chunking estructural: 15 presupuestos con 52
    # componentes dan exactamente 52 chunks, ni más ni menos.
    assert len(chunks) == sum(len(b.components) for b in budgets) == 52


def test_chunk_id_is_traceable_and_unique(chunks):
    first = chunks[0]
    assert first.chunk_id == "BUD-2024-001::AUTH-001"
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_text_starts_with_parent_header_then_component(budgets, chunks):
    budget, component, text = budgets[0], budgets[0].components[0], chunks[0].text
    assert text.startswith(f"[Project: {budget.project_summary}]\n")
    assert f"[Client sector: finance | Year: 2024 | Main tech: {budget.main_technology}]" in text
    assert f"Component: {component.name}" in text
    assert f"Description: {component.description}" in text
    assert "Tech stack: ruby_on_rails, postgresql, redis" in text


def test_metadata_has_filterable_fields_but_client_name_is_not_embedded(budgets, chunks):
    assert set(chunks[0].metadata) == {
        "budget_id",
        "component_id",
        "client_sector",
        "main_technology",
        "year",
        "complexity",
        "estimated_hours",
    }
    # NOTA: el nombre del cliente no aparece en el texto embebido: no aporta
    # significado para comparar trabajos y es un dato sensible (Sesión 6, PII).
    assert budgets[0].client_metadata.name not in chunks[0].text


def test_token_count_matches_tiktoken(chunks):
    for chunk in chunks:
        assert chunk.token_count == count_tokens(chunk.text) > 0
