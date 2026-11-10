"""Render versioned Jinja2 prompt templates into (system, user) message pairs.

Layout: ``app/prompts/<task>/<version>/{system,user,examples}.j2``. A published
version is immutable: changing a prompt means adding ``v2/`` next to ``v1/``,
never editing ``v1/`` in place, so every logged ``prompt_version`` keeps
pointing at the exact text that produced a response.

NOTA: ¿por qué un loader y no un f-string?
-----------------------------------------
Antes (sesión 2-3) el prompt era un f-string dentro de Python:

    system = f"You are a consultant... {examples}... {format_rules}"

Problemas de ese enfoque:
  1. Para cambiar una coma del prompt hay que tocar código y redesplegar.
  2. El prompt se mezcla con la lógica: cuesta leerlo entero de un vistazo.
  3. No hay historial: si el prompt cambia, no sabes qué texto exacto produjo
     una respuesta de la semana pasada.

Con este módulo, el prompt vive en archivos ``.j2`` (texto plano con
``{{ variables }}`` y ``{% if %}``) y Python solo hace tres cosas:
elegir la versión, pasarle los datos del formulario y devolver el texto.

Flujo completo de una petición:

    EstimationRequest ──► render_estimation_prompt(request, "v1")
                              │
                              ├─ estimation/v1/system.j2  (+ include examples.j2)
                              └─ estimation/v1/user.j2
                              │
                              ▼
                        (system, user) ──► LLMWrapper ──► modelo

Sesión 5 añade dos funciones más sobre el mismo ``_render``:

    render_session_prompt()     → estimation/v3 con <project_metadata> y adjuntos
    render_extraction_prompt()  → extraction/v1 (el "LLM extractor" de hechos)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import EstimationRequest
from app.schemas.session import SessionTurn
from app.services.attachments import ExtractedAttachment
from app.services.sessions import ProjectMetadata

log = structlog.get_logger()

# NOTA: rutas calculadas a partir de este archivo (``__file__``), no del
# directorio desde el que se lanza el proceso. Así funciona igual con
# ``uvicorn`` en local, en Docker o desde pytest.
PROMPTS_DIR = Path(__file__).resolve().parent
ESTIMATION_DIR = PROMPTS_DIR / "estimation"
DEFAULT_VERSION = "v1"
# NOTA: la conversación usa siempre v3, la primera versión que sabe pintar
# <project_metadata> y los adjuntos. v1 y v2 no cambian (son inmutables).
SESSION_VERSION = "v3"
EXTRACTION_VERSION = "v1"

# NOTA: el Environment se crea UNA sola vez al importar el módulo. Jinja2
# cachea internamente los templates ya compilados, así que renderizar en cada
# petición es barato (no vuelve a leer ni a parsear los .j2 de disco).
_env = Environment(
    # Busca los templates por ruta relativa a app/prompts/,
    # p. ej. get_template("estimation/v1/system.j2").
    loader=FileSystemLoader(PROMPTS_DIR),
    # Fail loudly on a missing/typo'd variable instead of rendering "".
    # NOTA: sin esto, ``{{ descripton }}`` (typo) se renderiza como "" en
    # silencio y el modelo recibe un prompt sin la descripción... y nadie se
    # entera. Con StrictUndefined el render lanza UndefinedError al instante.
    undefined=StrictUndefined,
    # NOTA: trim_blocks y lstrip_blocks controlan el espacio en blanco. Sin
    # ellos, cada línea con solo ``{% if %}`` o ``{% endif %}`` deja una línea
    # vacía o espacios sueltos en el prompt final. Con ellos, las etiquetas de
    # control "desaparecen" del todo y el texto queda limpio.
    trim_blocks=True,  # quita el salto de línea justo después de un {% ... %}
    lstrip_blocks=True,  # quita los espacios/tabs antes de un {% ... %}
    keep_trailing_newline=False,
    # NOTA: autoescape es para HTML (convierte < en &lt;). Aquí generamos texto
    # para un LLM, y necesitamos que <project_description> salga tal cual.
    autoescape=False,
)


class UnknownPromptVersionError(ValueError):
    """Raised when a request asks for a prompt version that does not exist."""

    # NOTA: una excepción propia (en vez de un ValueError genérico) permite que
    # el router la capture de forma específica y la traduzca a un HTTP 422,
    # sin tragarse por error otros ValueError que sí serían bugs.


def available_versions() -> list[str]:
    """Return the estimation prompt versions found on disk, sorted (v1, v2, ...)."""
    # NOTA: la "fuente de verdad" de qué versiones existen es el sistema de
    # archivos. Añadir v3 = crear la carpeta estimation/v3/ con su system.j2;
    # no hay que tocar ninguna lista en Python.
    return sorted(
        p.name for p in ESTIMATION_DIR.iterdir() if p.is_dir() and (p / "system.j2").is_file()
    )


def render_estimation_prompt(
    request: EstimationRequest, version: str = DEFAULT_VERSION
) -> tuple[str, str]:
    """Render the estimation prompt for ``request`` and return ``(system, user)``.

    NOTA: la firma recibe ``version`` como parámetro con valor por defecto.
    El resto del código (router, tests) llama siempre a esta misma función;
    cambiar de prompt es cambiar un argumento, no reescribir el endpoint.

    Ejemplo::

        >>> req = EstimationRequest(description="App de reservas de pádel...",
        ...                         project_type="mobile_app",
        ...                         detail_level="summary",
        ...                         output_format="phases_table")
        >>> system, user = render_estimation_prompt(req, version="v2")
        >>> user.startswith("<project_description>")
        True
    """
    # Whitelisting against the directories on disk also rules out path traversal.
    # NOTA: si alguien manda ?prompt_version=../../etc, ese string nunca llega a
    # Jinja: solo se aceptan nombres que existan como carpeta en estimation/.
    if version not in available_versions():
        raise UnknownPromptVersionError(
            f"Unknown prompt version '{version}'. Available: {', '.join(available_versions())}"
        )

    # mode="json" turns the enums into their string values ("phases_table"),
    # which is what the templates compare against.
    # NOTA: model_dump() normal devolvería OutputFormat.PHASES_TABLE (un enum);
    # con mode="json" devuelve "phases_table" (un str). Por eso en los .j2
    # podemos escribir {% if output_format == "phases_table" %}.
    # El diccionario resultante es el "contexto" del template: cada clave pasa
    # a ser una variable disponible en el .j2 ({{ description }}, etc.).
    context = request.model_dump(mode="json")
    # v3+ templates also read the session state; a stateless call has none.
    # NOTA: sin sesión, v3 recibe la memoria y los adjuntos vacíos y se
    # comporta como v2 con el bloque <project_metadata> sin contenido.
    context.update(project_metadata=ProjectMetadata().model_dump(), attachments=[])
    return _render(
        "estimation",
        version,
        context,
        project_type=context["project_type"],
        detail_level=context["detail_level"],
        output_format=context["output_format"],
    )


def render_session_prompt(
    turn: SessionTurn,
    project_metadata: ProjectMetadata,
    attachments: Sequence[ExtractedAttachment] = (),
    version: str = SESSION_VERSION,
) -> tuple[str, str]:
    """Render one conversation turn: system with the project facts, user with attachments.

    The system prompt is regenerated on every turn from the *current* metadata,
    so the facts are always up to date even when old turns left the window.
    """
    # NOTA: el transcript se pasa como ``description`` para reutilizar el mismo
    # bloque <project_description> que las versiones anteriores.
    context: dict[str, Any] = {
        "description": turn.transcript,
        "project_type": turn.project_type.value,
        "detail_level": turn.detail_level.value,
        "output_format": turn.output_format.value,
        "project_metadata": project_metadata.model_dump(),
        "attachments": [{"filename": a.filename, "text": a.text} for a in attachments],
    }
    return _render(
        "estimation",
        version,
        context,
        project_type=context["project_type"],
        detail_level=context["detail_level"],
        output_format=context["output_format"],
        attachments=len(attachments),
        has_metadata=not project_metadata.is_empty(),
    )


def render_extraction_prompt(
    known_facts: ProjectMetadata, user_message: str, assistant_answer: str
) -> tuple[str, str]:
    """Render the prompt that asks the LLM for the project facts of one turn as JSON."""
    context = {
        "known_facts": json.dumps(known_facts.model_dump(), ensure_ascii=False, indent=2),
        "user_message": user_message,
        "assistant_answer": assistant_answer,
    }
    return _render("extraction", EXTRACTION_VERSION, context)


def _render(task: str, version: str, context: dict[str, Any], **log_fields: Any) -> tuple[str, str]:
    """Render ``<task>/<version>/{system,user}.j2`` and log the prompt fingerprint."""
    # NOTA: dos templates, dos mensajes. El system lleva las instrucciones
    # (quién eres, cómo responder); el user lleva SOLO los datos del usuario.
    # Mantenerlos separados ayuda al modelo a distinguir órdenes de datos y
    # reduce el riesgo de que un texto del usuario se "cuele" como instrucción.
    system = _env.get_template(f"{task}/{version}/system.j2").render(**context).strip()
    user = _env.get_template(f"{task}/{version}/user.j2").render(**context).strip()

    # NOTA (bonus de observabilidad): el hash es una "huella" corta del prompt
    # exacto. Si en producción dos respuestas tienen el mismo prompt_hash, el
    # modelo recibió exactamente el mismo texto; si cambia, algo del prompt
    # cambió (versión o parámetros). Útil para depurar sin loggear el prompt
    # entero (que puede ser largo o contener datos del cliente).
    log.info(
        "prompt_rendered",
        prompt=task,
        prompt_version=version,
        prompt_hash=hashlib.sha256(f"{system}\n\n{user}".encode()).hexdigest()[:12],
        system_chars=len(system),
        user_chars=len(user),
        **log_fields,
    )
    return system, user
