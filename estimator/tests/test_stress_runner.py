"""Smoke test of the stress runner and of the turn_observed telemetry (Session 6).

The runner drives the real FastAPI app in-process with the LLM replaced by a
stub, so it checks the wiring (observation in the response, snapshot, CSV
columns, metrics) without network or tokens.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_llm_wrapper, get_session_store
from app.main import app
from app.services.sessions import SessionStore
from evals.stress import run as runner
from evals.stress.fixtures.build_pdfs import build_pdf, fixture_path
from evals.stress.scenarios import get_scenario

THIRTEEN_FIELDS = {
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
}


class _Stub:
    """Fake wrapper: fixed estimation; the extractor keeps the project name."""

    def complete_messages(self, messages: list[dict], **kwargs) -> dict:
        return _result(
            "| Fase | Horas |\n|---|---|\n| Build | 100 |", tokens_in=100 * len(messages)
        )

    def complete(self, *, system_prompt: str, user_message: str, **kwargs) -> dict:
        name = next((n for n in ("Nimbus", "Helios", "Atlas") if n in user_message), None)
        facts = {"project_name": name, "mentioned_technologies": [], "agreed_scope": None}
        return _result(json.dumps(facts), tokens_in=50)


def _result(text: str, tokens_in: int) -> dict:
    return {
        "estimation": text,
        "model": "gpt-4o-mini",
        "provider": "openai",
        "finish_reason": "stop",
        "usage": {"input_tokens": tokens_in, "output_tokens": 20, "total_tokens": tokens_in + 20},
        "latency_ms": 7,
        "cost_usd": 0.0001,
        "cache_hit": False,
    }


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_llm_wrapper] = lambda: _Stub()
    app.dependency_overrides[get_session_store] = lambda: STORE
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


STORE = SessionStore(max_turns=6)


class _RecordingLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def info(self, event: str, **fields) -> None:
        self.events.append((event, fields))

    error = warning = info


def test_each_turn_emits_turn_observed_with_the_thirteen_fields(client, monkeypatch) -> None:
    # NOTA: main.py configura structlog con cache_logger_on_first_use, así que
    # structlog.testing.capture_logs no ve un logger ya usado por otros tests.
    # Sustituir el logger del módulo es más robusto.
    recorder = _RecordingLog()
    monkeypatch.setattr("app.routers.sessions.log", recorder)
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    response = client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": "Un SaaS llamado Nimbus."},
    )
    events = [fields for name, fields in recorder.events if name == "turn_observed"]
    assert len(events) == 1
    assert THIRTEEN_FIELDS <= events[0].keys()
    observation = response.json()["observation"]
    assert observation["turn_index"] == 1
    assert observation["messages_in_window"] == 2  # system + new user
    assert observation["cache_hit_kind"] == "none"
    # The real bill of a turn includes the metadata extractor call.
    assert observation["turn_cost_usd"] == pytest.approx(0.0002)
    # The snapshot exposes the same observation.
    snapshot = client.get(f"/api/v1/sessions/{session_id}").json()
    assert snapshot["last_observation"] == observation


def test_runner_walks_a_scenario_and_measures_drift(client, tmp_path) -> None:
    if not fixture_path(5).exists():
        build_pdf(5, fixture_path(5))
    rows = runner.run_session(
        client, get_scenario("growing"), size_kb=5, repeat=0, config=runner.RunConfig(max_turns=8)
    )
    assert [r["turn_index"] for r in rows] == list(range(1, 9))
    assert not any(r["error"] for r in rows)
    # Turn 1 carries the attachment, later turns do not.
    assert rows[0]["attachments_total_chars"] > 5000
    assert rows[1]["attachments_total_chars"] == 0
    # The window caps at system + 5 previous turns (10 messages) + new user = 12.
    assert max(r["messages_in_window"] for r in rows) == 12
    # The project name lives in the metadata, so it survives after leaving the window.
    assert rows[0]["memory_drift_passed"] == ""
    assert all(r["memory_drift_passed"] is True for r in rows[1:])
    # The stub extractor keeps nothing else: by turn 8 early facts left the window.
    assert rows[7]["facts_recall"] < 1.0
    # The head marker of the PDF leaves the window after turn 6.
    assert rows[5]["attach_head_in_session"] is True
    assert rows[6]["attach_head_in_session"] is False

    output = tmp_path / "results.csv"
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=runner.CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    assert len(output.read_text().splitlines()) == 9
