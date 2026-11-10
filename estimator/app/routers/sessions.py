"""Conversational endpoints: create a session and estimate turn by turn.

NOTA: flujo de un turno (POST /sessions/{id}/estimate)
-----------------------------------------------------

    multipart: transcript + project_type/... + attachments[]
        │
        1. Buscar la sesión (404 si no existe o el servicio se reinició)
        2. Extraer el texto de cada adjunto            → attachments.py (415/422)
        3. Renderizar v3 con la memoria ACTUAL          → loader.render_session_prompt
        4. messages = [system] + ventana + [user nuevo] → ConversationHistory
        5. Llamar al LLM con todo el array              → wrapper.complete_messages (502)
        6. Guardar el turno en el historial (la ventana descarta el más viejo)
        7. Actualizar la memoria con el LLM extractor   → metadata_extractor.py
        │
        ▼
    SessionEstimateResponse (estimación + memoria + estado de la ventana)

El historial y la memoria solo se modifican si la llamada al LLM tuvo éxito:
un fallo del proveedor deja la sesión exactamente como estaba.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.dependencies import get_llm_wrapper, get_session_store
from app.prompts.loader import SESSION_VERSION, render_session_prompt
from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType
from app.schemas.session import (
    TRANSCRIPT_MAX_CHARS,
    AttachmentInfo,
    SessionCreated,
    SessionEstimateResponse,
    SessionState,
    SessionTurn,
    TurnView,
)
from app.services.attachments import (
    ExtractedAttachment,
    UnreadableAttachmentError,
    UnsupportedAttachmentError,
    extract_attachment,
)
from app.services.llm_wrapper import LLMWrapper
from app.services.metadata_extractor import extract_metadata
from app.services.sessions import Session, SessionNotFoundError, SessionStore

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


def _get_session(store: SessionStore, session_id: str) -> Session:
    """Look the session up, turning an unknown id into a 404."""
    # NOTA: como las sesiones viven en memoria, un 404 también aparece tras
    # reiniciar el servicio. El cliente debe crear una sesión nueva.
    try:
        return store.get(session_id)
    except SessionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Create one with POST /api/v1/sessions.",
        ) from None


@router.post("", response_model=SessionCreated, status_code=status.HTTP_201_CREATED)
async def create_session(store: SessionStore = Depends(get_session_store)) -> SessionCreated:
    """Create an empty conversation and return its ``session_id``."""
    session = store.create()
    log.info("session_created", session_id=session.session_id, sessions=len(store))
    return SessionCreated(session_id=session.session_id, max_turns=store.max_turns)


@router.get("/{session_id}", response_model=SessionState)
async def get_session(
    session_id: str, store: SessionStore = Depends(get_session_store)
) -> SessionState:
    """Return the memory and the stored history of a session (debugging aid)."""
    session = _get_session(store, session_id)
    return SessionState(
        session_id=session.session_id,
        turn_count=session.turn_count,
        max_turns=session.history.max_turns,
        project_metadata=session.metadata,
        history=[TurnView(user=t.user, assistant=t.assistant) for t in session.history.turns],
    )


# Sync on purpose (blocking LLM calls run in FastAPI's worker threads).
# NOTA: multipart/form-data. Con archivos no se puede mandar JSON en el body,
# así que cada parámetro tipado llega como un campo de formulario (Form) y los
# archivos como File. FastAPI sigue validando los enums y las longitudes igual
# que con JSON: un project_type inválido devuelve 422 antes de entrar aquí.
@router.post("/{session_id}/estimate", response_model=SessionEstimateResponse)
def estimate_in_session(
    session_id: str,
    transcript: str = Form(min_length=1, max_length=TRANSCRIPT_MAX_CHARS),
    project_type: ProjectType = Form(ProjectType.WEB_SAAS),
    detail_level: DetailLevel = Form(DetailLevel.MEDIUM),
    output_format: OutputFormat = Form(OutputFormat.PHASES_TABLE),
    attachments: list[UploadFile] | None = File(
        default=None, description="Optional PDF or Word (.docx) files."
    ),
    store: SessionStore = Depends(get_session_store),
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> SessionEstimateResponse:
    """Estimate within a session: history and project metadata update on every call."""
    session = _get_session(store, session_id)
    if not transcript.strip():
        raise HTTPException(status_code=422, detail="transcript: must not be blank")
    turn = SessionTurn(
        transcript=transcript.strip(),
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
    )
    extracted = _extract_all(attachments or [])

    # NOTA: todo lo que lee y escribe la sesión va dentro del lock (ver Session).
    with session.lock:
        system, user = render_session_prompt(turn, session.metadata, extracted)
        messages = session.history.to_messages_list(system, pending_user=user)
        log.info(
            "session_turn_started",
            session_id=session_id,
            turn=session.turn_count + 1,
            history_turns=sum(1 for m in messages if m["role"] == "user") - 1,
            attachments=len(extracted),
        )
        try:
            result = wrapper.complete_messages(messages)
        except Exception as exc:  # noqa: BLE001 — every provider failure maps to 502
            log.error("session_turn_failed", session_id=session_id, error=str(exc))
            raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc

        session.history.add_turn(user, result["estimation"])
        session.turn_count += 1
        session.metadata = extract_metadata(wrapper, session.metadata, user, result["estimation"])

        return SessionEstimateResponse(
            text=result["estimation"],
            prompt_version=SESSION_VERSION,
            model=result["model"],
            provider=result["provider"],
            finish_reason=result["finish_reason"],
            usage=result["usage"],
            latency_ms=result["latency_ms"],
            cache_hit=result.get("cache_hit", False),
            cost_usd=result.get("cost_usd", 0.0),
            session_id=session_id,
            turn=session.turn_count,
            turns_in_window=len(session.history),
            max_turns=session.history.max_turns,
            project_metadata=session.metadata,
            attachments=[
                AttachmentInfo(filename=a.filename, chars=a.chars, truncated=a.truncated)
                for a in extracted
            ],
        )


def _extract_all(uploads: list[UploadFile]) -> list[ExtractedAttachment]:
    """Extract every upload, mapping unsupported types to 415 and bad files to 422."""
    extracted = []
    for upload in uploads:
        # NOTA: en un endpoint ``def`` (síncrono) el archivo se lee con
        # ``upload.file.read()``; en uno ``async def`` sería ``await upload.read()``.
        data = upload.file.read()
        try:
            extracted.append(extract_attachment(upload.filename or "attachment", data))
        except UnsupportedAttachmentError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except UnreadableAttachmentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return extracted
