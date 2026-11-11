"""Update a session's ``ProjectMetadata`` after each turn with an LLM extractor.

A second, short LLM call reads the last turn and returns the project facts as
JSON. It is more robust than regexes over free-text estimations (team size or
agreed scope are hard to pattern-match) at the price of one extra call per
turn, kept cheap with a small ``max_tokens`` and the same exact-match cache.

NOTA: por qué un fallo aquí NO rompe la estimación
-------------------------------------------------
El usuario ya tiene su estimación cuando corre el extractor. Si el modelo
devuelve un JSON roto o el proveedor falla, se registra un warning y la
memoria se queda como estaba: se pierde un turno de hechos, no la respuesta.
"""

from __future__ import annotations

import json
import re

import structlog
from pydantic import ValidationError

from app.prompts.loader import render_extraction_prompt
from app.services.llm_wrapper import LLMWrapper
from app.services.sessions import ProjectMetadata

log = structlog.get_logger()

EXTRACTION_MAX_TOKENS = 400
# Models sometimes wrap JSON in ```json fences despite being told not to.
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_metadata(
    wrapper: LLMWrapper,
    current: ProjectMetadata,
    user_message: str,
    assistant_answer: str,
) -> ProjectMetadata:
    """Return ``current`` refined with the facts of the last turn (never raises)."""
    system, user = render_extraction_prompt(current, user_message, assistant_answer)
    try:
        result = wrapper.complete(
            system_prompt=system, user_message=user, max_tokens=EXTRACTION_MAX_TOKENS
        )
        update = parse_metadata(result["estimation"])
    except (ValueError, ValidationError) as exc:
        log.warning("metadata_extraction_invalid", error=str(exc))
        return current
    except Exception as exc:  # noqa: BLE001 — provider down: keep the old memory
        log.warning("metadata_extraction_failed", error=str(exc), error_type=type(exc).__name__)
        return current

    merged = current.merge(update)
    log.info(
        "metadata_extracted",
        project_name=merged.project_name,
        technologies=len(merged.mentioned_technologies),
        changed=merged != current,
    )
    return merged


def parse_metadata(raw: str) -> ProjectMetadata:
    """Parse the extractor's answer into a validated ``ProjectMetadata``."""
    # NOTA: dos capas de defensa. 1) json.loads comprueba que es JSON válido;
    # 2) model_validate comprueba tipos y rangos (p. ej. team size entre 1 y
    # 100). Si el modelo inventa claves extra, Pydantic las ignora.
    text = _FENCE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"No JSON object in extractor answer: {raw[:80]!r}")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Extractor answer is not a JSON object")
    data["mentioned_technologies"] = data.get("mentioned_technologies") or []
    return ProjectMetadata.model_validate(data)
