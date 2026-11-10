"""Conversation state per session: sliding-window history + project metadata.

The state lives in a plain dict inside the API process — no database, no Redis.
That volatility is accepted on purpose at this stage: a restart (or a ``--reload``
in dev) loses every conversation, and the service must run with a single worker
because each process would hold its own dict. Persistence arrives in a later
module; the in-memory store keeps the focus on *what* is remembered, not *where*.

NOTA: historial vs memoria
--------------------------
Son dos cosas distintas y por eso viven en dos estructuras:

  - **Historial** (``ConversationHistory``): los mensajes literales
    user/assistant. Es lo que viaja en el array ``messages`` de cada llamada.
    Crece con cada turno, así que se recorta con una ventana deslizante.
  - **Memoria** (``ProjectMetadata``): los HECHOS del proyecto (nombre, equipo,
    tecnologías, alcance). Es pequeña, no se recorta y se inyecta en el system
    prompt de cada turno. Aunque la ventana olvide el turno 1, el nombre del
    proyecto sigue en la memoria.

    Session
    ├── history   → [turno 1, turno 2, ... turno N]   (se descartan los viejos)
    └── metadata  → {project_name, team, technologies, scope}   (se acumula)
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, Field

DEFAULT_MAX_TURNS = 6


class ProjectMetadata(BaseModel):
    """Facts about the project under discussion, injected into the system prompt."""

    # NOTA: todos los campos son opcionales porque en el primer turno no se
    # sabe nada. Se van rellenando con lo que extrae el extractor tras cada
    # respuesta (app/services/metadata_extractor.py).
    project_name: str | None = None
    assumed_team_size: int | None = Field(default=None, ge=1, le=100)
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = None

    def is_empty(self) -> bool:
        """True while no fact has been captured yet (first turn of a session)."""
        return self == ProjectMetadata()

    def merge(self, update: ProjectMetadata) -> ProjectMetadata:
        """Return a new metadata where ``update`` refines ``self``.

        Scalar fields are overwritten only when ``update`` brings a value, so a
        turn that does not mention the project name never erases it.
        Technologies accumulate (case-insensitive, first spelling wins).
        """
        # NOTA: la memoria se ACUMULA. Si el turno 3 solo habla de pagos, el
        # extractor no devuelve el nombre del proyecto (None) y no debe borrarse.
        known = {t.lower() for t in self.mentioned_technologies}
        technologies = list(self.mentioned_technologies)
        for tech in update.mentioned_technologies:
            if tech.strip() and tech.lower() not in known:
                technologies.append(tech.strip())
                known.add(tech.lower())
        return ProjectMetadata(
            project_name=update.project_name or self.project_name,
            assumed_team_size=update.assumed_team_size or self.assumed_team_size,
            mentioned_technologies=technologies,
            agreed_scope=update.agreed_scope or self.agreed_scope,
        )


@dataclass(frozen=True)
class Turn:
    """One user message and the assistant answer it produced."""

    user: str
    assistant: str


class ConversationHistory:
    """Bounded user/assistant history with a sliding window of ``max_turns``.

    A turn is a user+assistant pair. The system prompt is not stored: it is
    regenerated on every call from the current project metadata and always
    placed first by ``to_messages_list``, so it can never fall out of the window.
    """

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.max_turns = max_turns
        # NOTA: deque(maxlen=N) es la ventana deslizante "gratis": al añadir el
        # turno N+1, Python descarta automáticamente el más antiguo.
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def add_turn(self, user: str, assistant: str) -> None:
        """Store a completed turn, dropping the oldest one beyond ``max_turns``."""
        self._turns.append(Turn(user=user, assistant=assistant))

    def to_messages_list(self, system_prompt: str, pending_user: str | None = None) -> list[dict]:
        """Build the ``messages`` array for the LLM API.

        With ``pending_user`` (the message being answered now), only the last
        ``max_turns - 1`` stored turns are included, so the call never carries
        more than ``max_turns`` user messages in total.
        """
        # NOTA: el turno en curso también cuenta. Con MAX_TURNS=6 se envían 5
        # turnos completos + la pregunta nueva = 6 mensajes de usuario.
        #
        #   [system] [u1][a1] [u2][a2] ... [u5][a5] [u_nuevo]
        #    siempre   ←────── ventana: últimos 5 ──────→  pendiente
        keep = self.max_turns - 1 if pending_user is not None else self.max_turns
        window = list(self._turns)[-keep:] if keep else []

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for turn in window:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        if pending_user is not None:
            messages.append({"role": "user", "content": pending_user})
        return messages


@dataclass
class Session:
    """State of one conversation: its history, its metadata and a lock."""

    session_id: str
    history: ConversationHistory
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    turn_count: int = 0
    # NOTA: dos peticiones simultáneas a la MISMA sesión podrían leer el mismo
    # historial y pisarse al guardar. El lock serializa los turnos de una
    # sesión; sesiones distintas siguen atendiéndose en paralelo.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class SessionNotFoundError(KeyError):
    """Raised when a request references a session id that does not exist."""


class SessionStore:
    """In-process registry of sessions keyed by ``session_id`` (volatile)."""

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        """Create an empty session with a random UUID v4 id."""
        # NOTA: uuid4 es aleatorio (122 bits): imposible de adivinar o de
        # repetir por accidente, a diferencia de un contador 1, 2, 3...
        session = Session(
            session_id=str(uuid.uuid4()),
            history=ConversationHistory(max_turns=self.max_turns),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        """Return the session or raise ``SessionNotFoundError`` (mapped to 404)."""
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFoundError(session_id) from None

    def __len__(self) -> int:
        return len(self._sessions)
