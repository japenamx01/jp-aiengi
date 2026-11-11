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


class SessionEstimateResponse(EstimationResponse):
    """An estimation plus the session state it left behind."""

    session_id: str
    turn: int = Field(description="1-based number of this turn in the session.")
    turns_in_window: int = Field(description="Completed turns kept in the sliding window.")
    max_turns: int
    project_metadata: ProjectMetadata
    attachments: list[AttachmentInfo] = Field(default_factory=list)


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
