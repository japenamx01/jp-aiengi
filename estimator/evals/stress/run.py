"""Stress runner of the CAG (block 5): scenarios × attachment sizes × repeats → CSV.

For every turn of every session it:

1. POSTs ``/api/v1/sessions/{id}/estimate`` (multipart) with the scripted
   transcript and, on turn 1, the calibrated PDF of the cell (none for 0 KB).
2. Reads ``observation`` from the response — the same ``TurnObservation`` the
   service logs as ``turn_observed``.
3. GETs ``/api/v1/sessions/{id}`` (history window + ProjectMetadata) and runs
   the three metrics plus the fact-tracker recall on that snapshot.
4. Writes one CSV row.

NOTA: por qué la observación viaja en la respuesta y no se parsea del log
-------------------------------------------------------------------------
Parsear ``docker compose logs | grep turn_observed`` obliga a casar cada línea
con su petición (por session_id y turno), depende del formato del renderer
(consola en dev, JSON en prod) y se rompe si el log rota. Devolver la misma
``TurnObservation`` en la respuesta (y en ``GET /sessions/{id}`` como
``last_observation``) es una línea más en el schema y deja el runner exacto.
El evento de log sigue existiendo para la observabilidad en producción.

NOTA: dos decisiones que afectan a los números
-----------------------------------------------
- Cache-bust (por defecto activo): la caché exacta de Redis usa como clave
  TODO el array de mensajes. Repetir un escenario sería un acierto de caché en
  cada turno (latencia ~0, coste 0) y las repeticiones no medirían nada. Por eso
  el turno 1 lleva una referencia única por sesión. ``--no-cache-bust`` lo
  desactiva para medir precisamente esa caché.
- El adjunto se envía solo en el turno 1 (como haría un usuario real), salvo
  ``--attach-every-turn``. Así se ve cuándo el documento sale de la ventana.

Usage::

    uv run python -m evals.stress.fixtures.build_pdfs
    uv run python -m evals.stress.run --http http://localhost:8000 \\
        --scenarios growing,pivot,contradiction --attachment-sizes 0,5,20,50,100 \\
        --repeats 3 --concurrency 5 --output evals/stress/results.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from evals.stress.fixtures.build_pdfs import HEAD_FACT, TAIL_FACT, fixture_path
from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
    haystack_of,
)
from evals.stress.scenarios import Scenario, get_scenario

API = "/api/v1/sessions"

OBSERVATION_COLUMNS = [
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    "prompt_chars",
    "history_turns",
    "model",
    "extractor_tokens_in",
    "extractor_tokens_out",
    "extractor_cost_usd",
    "extractor_latency_ms",
    "turn_cost_usd",
    "turn_latency_ms",
]

CSV_COLUMNS = [
    "scenario",
    "attachment_size_kb",
    "repeat",
    *OBSERVATION_COLUMNS,
    "wall_clock_ms",
    "latency_budget_passed",
    "cost_budget_passed",
    "memory_drift_passed",
    "tracked_fact",
    "facts_tracked",
    "facts_recall",
    "facts_recall_metadata",
    "facts_recall_history",
    "attach_head_in_answer",
    "attach_tail_in_answer",
    "attach_head_in_session",
    "superseded_in_metadata",
    "updated_in_session",
    "updated_in_answer",
    "metadata_project_name",
    "metadata_technologies",
    "error",
]


@dataclass(frozen=True)
class RunConfig:
    latency_budget_ms: int = 8000
    cost_budget_usd: float = 0.005
    cache_bust: bool = True
    attach_every_turn: bool = False
    max_turns: int | None = None


@contextmanager
def open_client(http_base_url: str | None) -> Iterator[Any]:
    """``--http`` → real httpx client; otherwise an in-process TestClient (smoke)."""
    if http_base_url:
        with httpx.Client(base_url=http_base_url.rstrip("/"), timeout=300.0) as client:
            yield client
        return
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client


def _attachment(size_kb: int) -> tuple[str, bytes] | None:
    if size_kb == 0:
        return None
    path = fixture_path(size_kb)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: run `uv run python -m evals.stress.fixtures.build_pdfs`"
        )
    return path.name, path.read_bytes()


def _contains(text: str, needle: str, aliases: tuple[str, ...] = ()) -> bool:
    lowered = text.lower()
    return any(n.lower() in lowered for n in (needle, *aliases) if n)


def _recall(metrics: list[MemoryDriftMetric], snapshot: dict) -> float | str:
    if not metrics:
        return ""
    return round(sum(m.evaluate(snapshot).score for m in metrics) / len(metrics), 4)


def run_session(
    client: Any,
    scenario: Scenario,
    size_kb: int,
    repeat: int,
    config: RunConfig,
) -> list[dict[str, Any]]:
    """Walk one scenario in a fresh session and return one row per turn."""
    attachment = _attachment(size_kb)
    latency_metric = LatencyBudgetMetric(config.latency_budget_ms)
    cost_metric = CostBudgetMetric(config.cost_budget_usd)
    first = scenario.turns[0]
    drift_metric = MemoryDriftMetric(first.fact, aliases=first.aliases)
    tag = f" (Ref. interna: stress-{uuid.uuid4().hex[:8]})" if config.cache_bust else ""

    created = client.post(API)
    created.raise_for_status()
    session_id = created.json()["session_id"]

    turns = scenario.turns[: config.max_turns] if config.max_turns else scenario.turns
    rows: list[dict[str, Any]] = []
    for index, turn in enumerate(turns, start=1):
        base = {
            "scenario": scenario.name,
            "attachment_size_kb": size_kb,
            "repeat": repeat,
            "turn_index": index,
            "session_id": session_id,
            "tracked_fact": first.fact,
        }
        data = {
            "transcript": turn.transcript + (tag if index == 1 else ""),
            "project_type": scenario.project_type.value,
            "detail_level": scenario.detail_level.value,
            "output_format": scenario.output_format.value,
        }
        files = None
        if attachment and (index == 1 or config.attach_every_turn):
            files = [("attachments", (attachment[0], attachment[1], "application/pdf"))]
        try:
            t0 = time.perf_counter()
            response = client.post(f"{API}/{session_id}/estimate", data=data, files=files)
            wall_ms = int((time.perf_counter() - t0) * 1000)
            response.raise_for_status()
            payload = response.json()
            snapshot_response = client.get(f"{API}/{session_id}")
            snapshot_response.raise_for_status()
            snapshot = snapshot_response.json()
        except Exception as exc:  # noqa: BLE001 — record it and abandon this session
            rows.append({**base, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            break  # later turns would measure a broken conversation

        observation = payload.get("observation")
        if observation is None:
            raise RuntimeError("response has no 'observation': is the service on session_6?")
        answer = payload.get("text", "")

        # Facts stated in turns 1..N-1 (deduplicated): what turn N should still know.
        previous = {t.fact: t for t in turns[: index - 1] if t.fact}
        everywhere = [MemoryDriftMetric(t.fact, aliases=t.aliases) for t in previous.values()]
        in_metadata = [
            MemoryDriftMetric(t.fact, where=("metadata",), aliases=t.aliases)
            for t in previous.values()
        ]
        in_history = [
            MemoryDriftMetric(t.fact, where=("history",), aliases=t.aliases)
            for t in previous.values()
        ]

        row = {**base, **{k: observation.get(k) for k in OBSERVATION_COLUMNS}}
        row.update(
            wall_clock_ms=wall_ms,
            latency_budget_passed=latency_metric.evaluate(observation).passed,
            cost_budget_passed=cost_metric.evaluate(observation).passed,
            memory_drift_passed="" if index == 1 else drift_metric.evaluate(snapshot).passed,
            facts_tracked=len(previous),
            facts_recall=_recall(everywhere, snapshot),
            facts_recall_metadata=_recall(in_metadata, snapshot),
            facts_recall_history=_recall(in_history, snapshot),
            metadata_project_name=(snapshot.get("project_metadata") or {}).get("project_name"),
            metadata_technologies=len(
                (snapshot.get("project_metadata") or {}).get("mentioned_technologies") or []
            ),
            error="",
        )
        if attachment:
            session_text = haystack_of(snapshot, ("metadata", "history"))
            row.update(
                attach_head_in_answer=_contains(answer, HEAD_FACT),
                attach_tail_in_answer=_contains(answer, TAIL_FACT),
                attach_head_in_session=_contains(session_text, HEAD_FACT),
            )
        change = scenario.supersession
        if change and index >= change.at_turn:
            metadata_text = haystack_of(snapshot, ("metadata",))
            session_text = haystack_of(snapshot, ("metadata", "history"))
            row.update(
                superseded_in_metadata=_contains(metadata_text, change.old, change.old_aliases),
                updated_in_session=_contains(session_text, change.new, change.new_aliases),
                updated_in_answer=_contains(answer, change.new, change.new_aliases),
            )
        rows.append(row)
    return rows


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def print_summary(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("error"):
            groups.setdefault((row["scenario"], int(row["attachment_size_kb"])), []).append(row)
    header = (
        f"{'scenario':<14}{'kb':>4}{'n':>5}{'P50 ms':>9}{'P95 ms':>9}"
        f"{'$/session':>11}{'drift %':>9}{'recall':>8}"
    )
    print("\n" + header + "\n" + "-" * len(header))
    for (scenario, kb), group in sorted(groups.items()):
        latencies = [float(r["latency_ms"]) for r in group]
        sessions = {r["session_id"] for r in group}
        cost = sum(float(r["turn_cost_usd"]) for r in group) / len(sessions)
        drifts = [r["memory_drift_passed"] for r in group if r["memory_drift_passed"] != ""]
        drift = 100 * sum(1 for d in drifts if d in (True, "True")) / len(drifts) if drifts else 0
        recalls = [float(r["facts_recall"]) for r in group if r["facts_recall"] != ""]
        print(
            f"{scenario:<14}{kb:>4}{len(group):>5}{_percentile(latencies, 50):>9.0f}"
            f"{_percentile(latencies, 95):>9.0f}{cost:>11.4f}{drift:>8.1f}%"
            f"{(statistics.mean(recalls) if recalls else 0):>8.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CAG stress runner (Session 6).")
    parser.add_argument("--http", default=None, help="Base URL, e.g. http://localhost:8000")
    parser.add_argument("--scenarios", default="growing,pivot,contradiction")
    parser.add_argument("--attachment-sizes", default="0,5,20,50,100")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1, help="Sessions run in parallel.")
    parser.add_argument("--max-turns", type=int, default=None, help="Cut scenarios (smoke).")
    parser.add_argument("--latency-budget-ms", type=int, default=8000)
    parser.add_argument("--cost-budget-usd", type=float, default=0.005)
    parser.add_argument("--no-cache-bust", action="store_true")
    parser.add_argument("--attach-every-turn", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("evals/stress/results.csv"))
    args = parser.parse_args(argv)

    scenarios = [get_scenario(name.strip()) for name in args.scenarios.split(",")]
    sizes = [int(s) for s in args.attachment_sizes.split(",")]
    for kb in sizes:  # fail before spending a single token
        _attachment(kb)
    config = RunConfig(
        latency_budget_ms=args.latency_budget_ms,
        cost_budget_usd=args.cost_budget_usd,
        cache_bust=not args.no_cache_bust,
        attach_every_turn=args.attach_every_turn,
        max_turns=args.max_turns,
    )
    cells = [(s, kb, r) for s in scenarios for kb in sizes for r in range(args.repeats)]
    print(f"Stress run: {len(cells)} sessions, concurrency {args.concurrency}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    started = time.perf_counter()
    with (
        open_client(args.http) as client,
        ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        futures = {
            pool.submit(run_session, client, s, kb, r, config): (s, kb, r) for s, kb, r in cells
        }
        for done, future in enumerate(as_completed(futures), start=1):
            scenario, kb, repeat = futures[future]
            session_rows = future.result()
            with lock:
                rows.extend(session_rows)
            errors = sum(1 for r in session_rows if r.get("error"))
            print(
                f"  [{done}/{len(cells)}] {scenario.name:<14} kb={kb:>3} repeat={repeat} "
                f"turns={len(session_rows)} errors={errors} "
                f"t={int(time.perf_counter() - started)}s"
            )

    rows.sort(
        key=lambda r: (r["scenario"], int(r["attachment_size_kb"]), r["repeat"], r["turn_index"])
    )
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.output} in {int(time.perf_counter() - started)}s")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
