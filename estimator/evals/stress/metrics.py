"""The three binary metrics of the stress test (block 4 of the exercise).

NOTA: por qué un módulo propio (``evals/stress/metrics.py``)
-----------------------------------------------------------
Este repo no tiene todavía el framework de evals de la clase
(``evals/metrics.py`` con ``MetricResult`` + ``run_all_metrics`` sobre un
``EstimationResult``). Estas métricas además no evalúan una estimación: leen
la telemetría de un turno (``TurnObservation``) y el snapshot de la sesión
(``GET /sessions/{id}``). Por eso viven aquí y definen su propio
``MetricResult`` con el mismo shape (``name``, ``score``, ``passed``,
``details``), para que el día que llegue el framework general se puedan mover
sin cambiar a quien las usa.

Determinismo > sofisticación: nada de embeddings ni LLM-as-judge. El drift es
un "contiene" case-insensitive sobre el snapshot. Cuando dice que el hecho no
está, el hecho no está.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

# Where MemoryDriftMetric looks inside a GET /sessions/{id} snapshot.
#   project_name / technologies / scope / team_size → one ProjectMetadata field
#   metadata → the whole ProjectMetadata (the long-term "memory")
#   history  → the literal user/assistant messages still in the sliding window
#   any      → the whole snapshot JSON
Where = Literal["project_name", "technologies", "scope", "team_size", "metadata", "history", "any"]
DEFAULT_WHERE: tuple[Where, ...] = ("metadata", "history")


@dataclass(frozen=True)
class MetricResult:
    """Outcome of one metric on one turn."""

    name: str
    score: float
    passed: bool
    details: str = ""


def _field(observation: Any, name: str) -> Any:
    """Read ``name`` from a TurnObservation, a dict or a CSV row alike."""
    if isinstance(observation, Mapping):
        return observation[name]
    return getattr(observation, name)


class LatencyBudgetMetric:
    """1.0 if the turn's LLM latency is within ``budget_ms``; 0.0 otherwise.

    NOTA: un presupuesto convierte el SLA en un test. "La estimación tarda
    menos de 8 s" deja de ser una impresión y pasa a ser una fila verde/roja.
    """

    name = "latency_budget"

    def __init__(self, budget_ms: int, field: str = "latency_ms") -> None:
        if budget_ms <= 0:
            raise ValueError("budget_ms must be positive")
        self.budget_ms = budget_ms
        self.field = field

    def evaluate(self, observation: Any) -> MetricResult:
        latency = int(_field(observation, self.field))
        passed = latency <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"{self.field}={latency} ms vs budget {self.budget_ms} ms",
        )


class CostBudgetMetric:
    """1.0 if the cost of ONE turn is within ``budget_usd``; 0.0 otherwise.

    By default it reads ``turn_cost_usd`` (estimation + metadata extractor), the
    real bill of a turn; pass ``field="cost_usd"`` for the estimation call only.
    The accumulated cost of a session is computed by the analysis, not here.
    """

    name = "cost_budget"

    def __init__(self, budget_usd: float, field: str = "turn_cost_usd") -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        self.budget_usd = budget_usd
        self.field = field

    def evaluate(self, observation: Any) -> MetricResult:
        cost = float(_field(observation, self.field))
        passed = cost <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"{self.field}=${cost:.6f} vs budget ${self.budget_usd:.6f}",
        )


class MemoryDriftMetric:
    """1.0 if a fact stated at turn k still appears in the session at turn N > k.

    ``fact`` is matched as a case-insensitive substring; ``aliases`` accept other
    spellings the model may use ("80000" / "80.000" / "80k"). ``where`` narrows
    the search to parts of the snapshot (see ``Where``). In this CAG there is no
    summary and no anchors: a fact survives either in ``ProjectMetadata`` or
    because its turn (or an answer that repeats it) is still in the window.
    """

    name = "memory_drift"

    def __init__(
        self,
        fact: str,
        where: Iterable[Where] = DEFAULT_WHERE,
        aliases: Iterable[str] = (),
    ) -> None:
        if not fact or not fact.strip():
            raise ValueError("fact must be a non-empty string")
        self.fact = fact
        self.where = tuple(where)
        if not self.where:
            raise ValueError("where must name at least one place to look")
        self.needles = [n.lower() for n in (fact, *aliases) if n.strip()]

    def evaluate(self, snapshot: Mapping[str, Any]) -> MetricResult:
        haystack = haystack_of(snapshot, self.where).lower()
        found = next((n for n in self.needles if n in haystack), None)
        return MetricResult(
            name=self.name,
            score=1.0 if found else 0.0,
            passed=found is not None,
            details=(
                f"fact={self.fact!r} where={','.join(self.where)} "
                + (f"present as {found!r}" if found else "missing")
            ),
        )


def haystack_of(snapshot: Mapping[str, Any], where: Iterable[Where]) -> str:
    """Concatenate the slices of a ``GET /sessions/{id}`` snapshot named in ``where``."""
    metadata = snapshot.get("project_metadata") or {}
    parts: list[str] = []
    for place in where:
        if place == "project_name":
            parts.append(str(metadata.get("project_name") or ""))
        elif place == "technologies":
            parts.append(" | ".join(metadata.get("mentioned_technologies") or []))
        elif place == "scope":
            parts.append(str(metadata.get("agreed_scope") or ""))
        elif place == "team_size":
            parts.append(str(metadata.get("assumed_team_size") or ""))
        elif place == "metadata":
            parts.append(json.dumps(metadata, ensure_ascii=False))
        elif place == "history":
            for turn in snapshot.get("history") or []:
                parts.append(turn.get("user", ""))
                parts.append(turn.get("assistant", ""))
        elif place == "any":
            parts.append(json.dumps(snapshot, ensure_ascii=False, default=str))
        else:
            raise ValueError(f"unknown place {place!r}")
    return "\n".join(parts)
