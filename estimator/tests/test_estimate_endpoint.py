from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_llm_wrapper
from app.main import app

VALID_BODY = {
    "description": "App móvil para que los vecinos reporten baches con foto y ubicación.",
    "project_type": "mobile_app",
    "detail_level": "summary",
    "output_format": "line_items",
}


class _StubWrapper:
    """Records the messages it receives and returns a canned completion."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    def complete(self, *, system_prompt: str, user_message: str, **kwargs) -> dict:
        self.calls.append({"system_prompt": system_prompt, "user_message": user_message})
        if self.fail:
            raise RuntimeError("provider down")
        return {
            "estimation": "- Backend — 40 h — 2,500 EUR",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280},
            "latency_ms": 900,
            "cost_usd": 0.000228,
            "cache_hit": False,
        }


@pytest.fixture
def stub() -> Iterator[_StubWrapper]:
    wrapper = _StubWrapper()
    app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    yield wrapper
    app.dependency_overrides.pop(get_llm_wrapper, None)


def test_estimate_returns_text_and_prompt_version(client: TestClient, stub: _StubWrapper) -> None:
    response = client.post("/api/v1/estimate", json=VALID_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "- Backend — 40 h — 2,500 EUR"
    assert body["prompt_version"] == "v1"
    assert body["model"] == "gpt-4o-mini"
    assert body["usage"]["input_tokens"] == 1200


def test_estimate_sends_system_and_user_as_separate_messages(
    client: TestClient, stub: _StubWrapper
) -> None:
    client.post("/api/v1/estimate", json=VALID_BODY)

    call = stub.calls[0]
    assert call["system_prompt"].startswith("You are a senior software consultant")
    assert VALID_BODY["description"] in call["user_message"]
    assert VALID_BODY["description"] not in call["system_prompt"]


def test_estimate_accepts_prompt_version_query_param(
    client: TestClient, stub: _StubWrapper
) -> None:
    response = client.post("/api/v1/estimate?prompt_version=v2", json=VALID_BODY)

    assert response.status_code == 200
    assert response.json()["prompt_version"] == "v2"
    assert "executive summary" in stub.calls[0]["system_prompt"]


def test_estimate_rejects_unknown_prompt_version(client: TestClient, stub: _StubWrapper) -> None:
    response = client.post("/api/v1/estimate?prompt_version=v99", json=VALID_BODY)

    assert response.status_code == 422
    assert "v99" in response.json()["detail"]
    assert stub.calls == []


@pytest.mark.parametrize(
    "override",
    [
        {"description": "muy corto"},  # < 20 chars
        {"description": "x" * 2001},  # > 2000 chars
        {"project_type": "MOBILE_APP"},  # enum names are not valid values
        {"output_format": "pdf"},
    ],
)
def test_estimate_validates_the_form_contract(
    client: TestClient, stub: _StubWrapper, override: dict
) -> None:
    response = client.post("/api/v1/estimate", json={**VALID_BODY, **override})

    assert response.status_code == 422
    assert stub.calls == []


def test_estimate_maps_provider_failure_to_502(client: TestClient) -> None:
    app.dependency_overrides[get_llm_wrapper] = lambda: _StubWrapper(fail=True)
    try:
        response = client.post("/api/v1/estimate", json=VALID_BODY)
    finally:
        app.dependency_overrides.pop(get_llm_wrapper, None)

    assert response.status_code == 502


def test_prompt_preview_returns_rendered_messages_without_calling_llm(
    client: TestClient, stub: _StubWrapper
) -> None:
    response = client.post("/api/v1/prompt/preview?prompt_version=v2", json=VALID_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["prompt_version"] == "v2"
    assert "<project_description>" in body["user"]
    assert "## Output format: line_items" in body["system"]
    assert stub.calls == []


def test_list_prompt_versions(client: TestClient) -> None:
    response = client.get("/api/v1/prompts/versions")

    assert response.status_code == 200
    assert response.json()[:2] == ["v1", "v2"]
