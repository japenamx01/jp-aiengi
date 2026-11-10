"""Unit tests of the LLM metadata extractor: parsing, merge and failure tolerance."""

import json

import pytest

from app.services.metadata_extractor import extract_metadata, parse_metadata
from app.services.sessions import ProjectMetadata


class _ExtractorStub:
    def __init__(self, answer: str | Exception) -> None:
        self.answer = answer
        self.prompts: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_message: str, **kwargs) -> dict:
        self.prompts.append((system_prompt, user_message))
        if isinstance(self.answer, Exception):
            raise self.answer
        return {"estimation": self.answer}


def test_parse_accepts_json_inside_markdown_fences() -> None:
    raw = '```json\n{"project_name": "VetCare", "mentioned_technologies": ["Flutter"]}\n```'
    assert parse_metadata(raw) == ProjectMetadata(
        project_name="VetCare", mentioned_technologies=["Flutter"]
    )


def test_parse_treats_null_technologies_as_empty() -> None:
    assert parse_metadata('{"mentioned_technologies": null}').mentioned_technologies == []


@pytest.mark.parametrize(
    "raw", ["no json here", "[1, 2]", '{"assumed_team_size": "a lot"}', '{"assumed_team_size": 0}']
)
def test_parse_rejects_invalid_answers(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_metadata(raw)


def test_extract_merges_the_answer_into_the_current_memory() -> None:
    current = ProjectMetadata(project_name="VetCare", mentioned_technologies=["React"])
    stub = _ExtractorStub(json.dumps({"project_name": None, "mentioned_technologies": ["Stripe"]}))

    result = extract_metadata(stub, current, "Add Stripe payments", "Payments: 24 h")

    assert result.project_name == "VetCare"
    assert result.mentioned_technologies == ["React", "Stripe"]
    # The extractor prompt carries the known facts and the whole turn.
    system, user = stub.prompts[0]
    assert "Return ONLY a JSON object" in system
    assert '"project_name": "VetCare"' in user
    assert "Add Stripe payments" in user and "Payments: 24 h" in user


# NOTA: el extractor nunca rompe el turno: con JSON roto o el proveedor caído
# devuelve la memoria anterior tal cual.
@pytest.mark.parametrize("answer", ["not json", RuntimeError("provider down")])
def test_extract_keeps_the_memory_when_it_fails(answer) -> None:
    current = ProjectMetadata(project_name="VetCare")
    assert extract_metadata(_ExtractorStub(answer), current, "u", "a") is current
