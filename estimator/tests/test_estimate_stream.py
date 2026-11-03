import json
from collections.abc import Iterator

from fastapi.testclient import TestClient

from app.dependencies import get_llm_wrapper
from app.main import app

VALID_BODY = {
    "description": "Pipeline nocturno que consolida ventas de tres tiendas en BigQuery.",
    "project_type": "data_pipeline",
    "detail_level": "medium",
    "output_format": "phases_table",
}


class _StubWrapper:
    """Minimal stand-in for LLMWrapper that yields canned chunks."""

    def __init__(self, chunks: list[str], metrics: dict | None = None) -> None:
        self._chunks = chunks
        self._metrics = metrics or {}
        self.calls: list[dict] = []

    def complete_stream(
        self,
        *,
        system_prompt: str,
        user_message: str,
        metrics: dict | None = None,
    ) -> Iterator[str]:
        self.calls.append({"system_prompt": system_prompt, "user_message": user_message})
        for chunk in self._chunks:
            yield chunk
        if metrics is not None:
            metrics.update(self._metrics)


def test_stream_endpoint_emits_token_and_done_events() -> None:
    stub = _StubWrapper(chunks=["Hello ", "from ", "the ", "estimator."])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/v1/estimate/stream",
                json=VALID_BODY,
            ) as response:
                assert response.status_code == 200
                body = b"".join(response.iter_bytes()).decode()
        # The response is one or more SSE messages separated by blank lines.
        assert "event: token" in body
        assert "data: Hello" in body
        assert "data: estimator." in body
        assert "event: done" in body
        assert len(stub.calls) == 1
        assert VALID_BODY["description"] in stub.calls[0]["user_message"]
        assert stub.calls[0]["system_prompt"].startswith("You are a senior software consultant")
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


def test_stream_endpoint_serialises_multiline_chunk_as_multiple_data_lines() -> None:
    """A chunk with internal newlines must be split into one ``data:`` line per
    physical line, with the message terminated by a blank line. Clients must
    join those data lines with ``\\n`` to recover the original payload — that
    is the contract this test pins down on the server side.
    """
    multiline_chunk = "## Project summary\n\nThis is the body."
    stub = _StubWrapper(chunks=[multiline_chunk])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/v1/estimate/stream",
                json=VALID_BODY,
            ) as response:
                assert response.status_code == 200
                body = b"".join(response.iter_bytes()).decode()
        assert "data: ## Project summary" in body
        assert "data: This is the body." in body
        assert "event: token" in body
        assert "event: done" in body
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


def test_stream_endpoint_rejects_short_description() -> None:
    stub = _StubWrapper(chunks=["irrelevant"])
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/estimate/stream",
                json={**VALID_BODY, "description": "too short"},
            )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)


def test_stream_endpoint_emits_metrics_before_done() -> None:
    metrics = {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "usage": {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500},
        "latency_ms": 4200,
        "cost_usd": 0.00036,
        "cache_hit": False,
    }
    stub = _StubWrapper(chunks=["Hello"], metrics=metrics)
    app.dependency_overrides[get_llm_wrapper] = lambda: stub
    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/api/v1/estimate/stream",
                json=VALID_BODY,
            ) as response:
                body = b"".join(response.iter_bytes()).decode()
        assert "event: metrics" in body
        assert body.index("event: metrics") < body.index("event: done")
        data_line = next(line for line in body.splitlines() if line.startswith("data: {"))
        assert json.loads(data_line[len("data: ") :]) == {**metrics, "prompt_version": "v1"}
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)
