"""Integration tests of the conversational endpoints (exercise step 7).

They drive the real FastAPI app through ``httpx.AsyncClient`` + ``ASGITransport``
(no server, no network) with the LLM replaced by a stub that records every
``messages`` array it receives.

NOTA: qué se puede comprobar con un stub y qué no
------------------------------------------------
El stub no "entiende" nada: devuelve respuestas fijas. Lo que sí prueban estos
tests es el CABLEADO de la memoria: qué mensajes llegan al LLM en cada turno,
que la metadata se acumula, que el texto del PDF entra en el prompt y que la
ventana nunca supera MAX_TURNS. Si el modelo real usa bien esa información es
otra pregunta (evals), igual que con los tests de template de la sesión 4.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest

from app.dependencies import get_llm_wrapper, get_session_store
from app.main import app
from app.services.sessions import SessionStore
from tests.helpers import make_docx, make_pdf

# NOTA: el plugin de pytest de anyio (viene con anyio) ejecuta los tests
# ``async def`` marcados así. anyio_backend elige asyncio como event loop.
pytestmark = pytest.mark.anyio

MAX_TURNS = 6
KNOWN_TECHNOLOGIES = ("React Native", "PostgreSQL", "Stripe", "Azure AD")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _ConversationStub:
    """Fake LLMWrapper: canned estimations plus a rule-based fake extractor."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.estimate_calls: list[list[dict]] = []

    def complete_messages(self, messages: list[dict], **kwargs) -> dict:
        # NOTA: guardamos una COPIA: la lista de mensajes la construye el
        # historial y no queremos que un turno posterior altere lo registrado.
        self.estimate_calls.append([dict(m) for m in messages])
        if self.fail:
            raise RuntimeError("provider down")
        # The answer reacts to the SSO requirement that only the PDF contains.
        hours = 200 if "Single Sign-On" in messages[-1]["content"] else 120
        return _result(f"| Phase | Hours |\n|---|---|\n| Build | {hours} |\n\nTotal hours: {hours}")

    def complete(self, *, system_prompt: str, user_message: str, **kwargs) -> dict:
        # Extractor call: pretend to be an LLM that reads the last turn only.
        turn = user_message.split("<user_message>", 1)[1]
        name = re.search(r"project (?:called|named) (\w+)", turn, re.IGNORECASE)
        facts = {
            "project_name": name.group(1) if name else None,
            "assumed_team_size": None,
            "mentioned_technologies": [t for t in KNOWN_TECHNOLOGIES if t in turn],
            "agreed_scope": None,
        }
        return _result(json.dumps(facts))


def _result(text: str) -> dict:
    return {
        "estimation": text,
        "model": "gpt-4o-mini",
        "provider": "openai",
        "finish_reason": "stop",
        "usage": {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050},
        "latency_ms": 10,
        "cost_usd": 0.0002,
        "cache_hit": False,
    }


@pytest.fixture
def stub() -> Iterator[_ConversationStub]:
    # NOTA: dos overrides: el LLM (stub) y el almacén de sesiones. Un store
    # nuevo por test garantiza que ningún test ve sesiones de otro.
    wrapper = _ConversationStub()
    store = SessionStore(max_turns=MAX_TURNS)
    app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    app.dependency_overrides[get_session_store] = lambda: store
    yield wrapper
    app.dependency_overrides.clear()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _new_session(client: httpx.AsyncClient) -> str:
    response = await client.post("/api/v1/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


async def _turn(client: httpx.AsyncClient, session_id: str, transcript: str, files=None):
    # NOTA: data= manda campos de formulario y files= los archivos; httpx
    # construye el multipart/form-data igual que un navegador.
    return await client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data={"transcript": transcript, "output_format": "phases_table"},
        files=files,
    )


# --- Required by the exercise ----------------------------------------------------------


async def test_two_turns_update_project_metadata(client, stub) -> None:
    session_id = await _new_session(client)

    first = await _turn(
        client, session_id, "A project called VetCare: clinic bookings built with React Native."
    )
    assert first.status_code == 200
    assert first.json()["project_metadata"]["project_name"] == "VetCare"
    assert first.json()["project_metadata"]["mentioned_technologies"] == ["React Native"]

    second = await _turn(client, session_id, "Add card payments with Stripe.")
    metadata = second.json()["project_metadata"]

    # The name survives a turn that does not mention it; technologies accumulate.
    assert metadata["project_name"] == "VetCare"
    assert metadata["mentioned_technologies"] == ["React Native", "Stripe"]
    # And the second call already carried the facts of the first in its system prompt.
    system_of_second_call = stub.estimate_calls[1][0]["content"]
    assert "- Project name: VetCare" in system_of_second_call
    assert second.json()["turn"] == 2


async def test_pdf_attachment_changes_the_estimation(client, stub) -> None:
    transcript = "A project called Portal: internal portal for 300 employees."
    without = await _turn(client, await _new_session(client), transcript)

    pdf = make_pdf("Technical specification", "Login must use Single Sign-On (SSO) with Azure AD.")
    with_pdf = await _turn(
        client,
        await _new_session(client),
        transcript,
        files=[("attachments", ("spec.pdf", pdf, "application/pdf"))],
    )

    assert with_pdf.status_code == 200
    # The extracted text reached the prompt, between the attachment markers...
    user_message = stub.estimate_calls[-1][-1]["content"]
    assert "--- attachment: spec.pdf ---" in user_message
    assert "Single Sign-On (SSO) with Azure AD" in user_message
    # ...and changed a concrete field of the output.
    assert "Total hours: 120" in without.json()["text"]
    assert "Total hours: 200" in with_pdf.json()["text"]
    assert with_pdf.json()["attachments"][0]["filename"] == "spec.pdf"
    assert "Azure AD" in with_pdf.json()["project_metadata"]["mentioned_technologies"]


async def test_history_sent_to_the_llm_never_exceeds_max_turns(client, stub) -> None:
    session_id = await _new_session(client)

    for n in range(1, 9):
        response = await _turn(client, session_id, f"Turn {n}: refine the scope.")
        assert response.status_code == 200

    for messages in stub.estimate_calls:
        user_messages = [m for m in messages if m["role"] == "user"]
        # NOTA: el system prompt siempre va primero, y como mucho MAX_TURNS
        # mensajes de usuario (5 turnos guardados + el actual).
        assert messages[0]["role"] == "system"
        assert sum(m["role"] == "system" for m in messages) == 1
        assert len(user_messages) <= MAX_TURNS

    last_call = stub.estimate_calls[-1]
    last_users = [m["content"] for m in last_call if m["role"] == "user"]
    assert len(last_users) == MAX_TURNS
    assert "Turn 3:" in last_users[0]  # turns 1 and 2 left the window
    assert "Turn 8:" in last_users[-1]
    assert response.json()["turns_in_window"] == MAX_TURNS


# --- Contract and error paths ----------------------------------------------------------


async def test_docx_attachment_is_extracted(client, stub) -> None:
    docx = make_docx("Integrations", table=[["System", "Protocol"], ["SAP", "REST"]])
    response = await _turn(
        client,
        await _new_session(client),
        "Internal tool to sync orders.",
        files=[("attachments", ("notes.docx", docx, "application/octet-stream"))],
    )

    assert response.status_code == 200
    assert "SAP | REST" in stub.estimate_calls[-1][-1]["content"]


async def test_unknown_session_returns_404(client, stub) -> None:
    response = await _turn(client, "does-not-exist", "Hello")
    assert response.status_code == 404
    assert stub.estimate_calls == []


async def test_unsupported_attachment_returns_415(client, stub) -> None:
    response = await _turn(
        client,
        await _new_session(client),
        "See the attached file.",
        files=[("attachments", ("notes.txt", b"plain text", "text/plain"))],
    )
    assert response.status_code == 415
    assert stub.estimate_calls == []


async def test_blank_transcript_returns_422(client, stub) -> None:
    response = await _turn(client, await _new_session(client), "   ")
    assert response.status_code == 422


async def test_provider_failure_leaves_the_session_untouched(client, stub) -> None:
    session_id = await _new_session(client)
    await _turn(client, session_id, "A project called VetCare built with PostgreSQL.")

    stub.fail = True
    failed = await _turn(client, session_id, "Add Stripe payments.")
    assert failed.status_code == 502

    state = (await client.get(f"/api/v1/sessions/{session_id}")).json()
    assert state["turn_count"] == 1
    assert len(state["history"]) == 1
    assert state["project_metadata"]["mentioned_technologies"] == ["PostgreSQL"]
