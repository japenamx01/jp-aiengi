"""Unit tests of the stress metrics (Session 6): one pass, one fail, one edge case each."""

from __future__ import annotations

import pytest

from evals.stress.fixtures.build_pdfs import _HEAD, _PARAGRAPH, _TAIL
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric
from evals.stress.scenarios import MIN_TURNS, all_scenarios

SNAPSHOT = {
    "session_id": "abc",
    "project_metadata": {
        "project_name": "Nimbus",
        "assumed_team_size": 3,
        "mentioned_technologies": ["React", "Okta"],
        "agreed_scope": "Catálogo de proveedores y flujo de cotizaciones.",
    },
    "history": [
        {"user": "Presupuesto confirmado: 60.000 EUR.", "assistant": "| Fase | Horas |"},
    ],
}


# --- LatencyBudgetMetric -------------------------------------------------------------


def test_latency_within_budget_passes() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 1200})
    assert result.passed and result.score == 1.0
    assert result.name == "latency_budget"


def test_latency_over_budget_fails() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 9000})
    assert not result.passed and result.score == 0.0
    assert "9000" in result.details


def test_latency_exactly_at_budget_passes() -> None:
    # Edge: the budget is inclusive (≤), as the exercise states.
    assert LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 4000}).passed


def test_budgets_must_be_positive() -> None:
    with pytest.raises(ValueError):
        LatencyBudgetMetric(budget_ms=0)
    with pytest.raises(ValueError):
        CostBudgetMetric(budget_usd=0)


# --- CostBudgetMetric ------------------------------------------------------------------


def test_cost_within_budget_passes() -> None:
    assert CostBudgetMetric(budget_usd=0.005).evaluate({"turn_cost_usd": 0.001}).passed


def test_cost_over_budget_fails() -> None:
    result = CostBudgetMetric(budget_usd=0.005).evaluate({"turn_cost_usd": 0.02})
    assert not result.passed and result.score == 0.0


def test_cost_reads_the_configured_field_and_cache_hits_cost_nothing() -> None:
    # Edge: a cache hit reports 0.0, which is within any positive budget.
    metric = CostBudgetMetric(budget_usd=0.001, field="cost_usd")
    assert metric.evaluate({"cost_usd": 0.0, "turn_cost_usd": 9.9}).passed


# --- MemoryDriftMetric -----------------------------------------------------------------


def test_fact_in_metadata_passes_case_insensitive() -> None:
    result = MemoryDriftMetric("NIMBUS").evaluate(SNAPSHOT)
    assert result.passed and "present" in result.details


def test_fact_absent_everywhere_fails() -> None:
    result = MemoryDriftMetric("Flutter").evaluate(SNAPSHOT)
    assert not result.passed and "missing" in result.details


def test_alias_matches_a_different_spelling() -> None:
    assert not MemoryDriftMetric("60000").evaluate(SNAPSHOT).passed
    assert MemoryDriftMetric("60000", aliases=("60.000",)).evaluate(SNAPSHOT).passed


def test_where_narrows_the_search() -> None:
    # Edge: the budget lives only in the window, not in the long-term memory.
    in_history = MemoryDriftMetric("60.000", where=("history",))
    in_metadata = MemoryDriftMetric("60.000", where=("metadata",))
    assert in_history.evaluate(SNAPSHOT).passed
    assert not in_metadata.evaluate(SNAPSHOT).passed
    assert MemoryDriftMetric("Okta", where=("technologies",)).evaluate(SNAPSHOT).passed


def test_empty_snapshot_and_invalid_arguments() -> None:
    assert not MemoryDriftMetric("Nimbus").evaluate({}).passed
    with pytest.raises(ValueError):
        MemoryDriftMetric("  ")
    with pytest.raises(ValueError):
        MemoryDriftMetric("Nimbus", where=())


# --- Scenario and fixture sanity -------------------------------------------------------


def test_scenarios_are_long_enough_and_start_with_the_project_name() -> None:
    for scenario in all_scenarios():
        assert len(scenario.turns) >= MIN_TURNS
        assert scenario.turns[0].fact in scenario.turns[0].transcript


def test_no_tracked_fact_appears_in_the_pdf_fixtures() -> None:
    # Otherwise the attachment would "remember" facts nobody stated.
    pdf_text = (_HEAD + _PARAGRAPH + _TAIL).lower()
    for scenario in all_scenarios():
        for turn in scenario.turns:
            for needle in (turn.fact, *turn.aliases):
                assert needle.lower() not in pdf_text, (scenario.name, needle)
