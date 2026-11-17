"""HTTP contract of the conversational endpoints (Session 5).

NOTA: la respuesta de un turno HEREDA de ``EstimationResponse`` (sesión 4):
mismo texto, modelo, tokens y caché, más el estado de la sesión. Así el
cliente que ya sabía pintar una estimación no tiene que cambiar nada para
pintar la de un turno.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.estimation import DetailLevel, EstimationResponse, OutputFormat, ProjectType
from app.services.sessions import ProjectMetadata

# Follow-up turns can be short ("add Stripe payments"), unlike the 20-char
# minimum of a first description in the stateless form.
TRANSCRIPT_MAX_CHARS = 4000


class SessionTurn(BaseModel):
    """Typed parameters of one conversation turn (sent as multipart form fields)."""

    # NOTA: a diferencia de EstimationRequest, aquí basta 1 carácter: en una
    # conversación "¿y si lo hacemos solo para iOS?" es un turno válido.
    transcript: str = Field(min_length=1, max_length=TRANSCRIPT_MAX_CHARS)
    project_type: ProjectType = ProjectType.WEB_SAAS
    detail_level: DetailLevel = DetailLevel.MEDIUM
    output_format: OutputFormat = OutputFormat.PHASES_TABLE


class SessionCreated(BaseModel):
    """Answer of ``POST /sessions``."""

    session_id: str
    max_turns: int


class AttachmentInfo(BaseModel):
    """What the service extracted from one uploaded file."""

    filename: str
    chars: int
    truncated: bool


class TurnObservation(BaseModel):
    """Telemetry of one conversational turn, emitted as the ``turn_observed`` event (Session 6).

    NOTA: un único evento con TODO lo relevante del turno, en vez de cinco logs
    sueltos (session_turn_started, attachment_extracted, llm_call_completed,
    metadata_extracted...). Así el CSV del stress test sale de una pasada, sin
    reconciliar timestamps, y correlacionar ``messages_in_window`` con
    ``cost_usd`` es trivial: están en la misma fila.

    Los 13 campos del enunciado se mantienen aunque este CAG no tenga anclas,
    summarizer ni tier dinámico: ``anchors_count`` y ``summary_chars`` valen 0 y
    ``last_resolved_tier`` es None. Dejarlos explícitos hace el CSV comparable
    con el de la versión de la clase (y con el RAG del módulo 3).
    """

    # --- the 13 fields of block 1 ---
    turn_index: int = Field(description="1-based turn number within the session.")
    session_id: str
    enriched_transcript_chars: int = Field(
        description="Chars of the user message sent this turn: transcript + extracted attachments."
    )
    attachments_total_chars: int = Field(description="Extracted attachment chars (0 if none).")
    messages_in_window: int = Field(
        description="Messages sent to the LLM this turn: system + window + new user message."
    )
    anchors_count: int = Field(default=0, description="Always 0: this CAG has no anchors.")
    summary_chars: int = Field(default=0, description="Always 0: this CAG has no summarizer.")
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int = Field(description="LLM latency of the estimation call (server side).")
    cache_hit_kind: str = Field(description='"none" | "exact" (there is no semantic cache).')
    last_resolved_tier: str | None = Field(
        default=None, description="Always None: the model is fixed (no dynamic tier)."
    )
    # --- extra fields: what the 13 fields cannot see in this architecture ---
    prompt_chars: int = Field(description="Chars of the whole messages array (system + window).")
    history_turns: int = Field(description="Previous turns carried in the window this call.")
    model: str
    extractor_tokens_in: int = 0
    extractor_tokens_out: int = 0
    extractor_cost_usd: float = 0.0
    extractor_latency_ms: int = 0
    turn_cost_usd: float = Field(description="cost_usd + extractor_cost_usd: the real turn bill.")
    turn_latency_ms: int = Field(description="Server time of the whole turn, extractor included.")


class SessionEstimateResponse(EstimationResponse):
    """An estimation plus the session state it left behind."""

    session_id: str
    turn: int = Field(description="1-based number of this turn in the session.")
    turns_in_window: int = Field(description="Completed turns kept in the sliding window.")
    max_turns: int
    project_metadata: ProjectMetadata
    attachments: list[AttachmentInfo] = Field(default_factory=list)
    observation: TurnObservation | None = Field(
        default=None, description="Telemetry of this turn (the turn_observed event)."
    )


class TurnView(BaseModel):
    user: str
    assistant: str


class SessionState(BaseModel):
    """Answer of ``GET /sessions/{session_id}``: memory and window, for debugging."""

    session_id: str
    turn_count: int
    max_turns: int
    project_metadata: ProjectMetadata
    history: list[TurnView]
    last_observation: TurnObservation | None = None
