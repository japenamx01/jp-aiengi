"""HTTP layer: turns typed form requests into LLM calls.

NOTA: este archivo es "pegamento". No construye prompts (eso es el loader) ni
habla con OpenAI/Anthropic (eso es el wrapper). Cada endpoint hace siempre los
mismos tres pasos:

    1. FastAPI valida el body contra EstimationRequest (si falla → 422 solo).
    2. render_estimation_prompt() convierte el request en (system, user).
    3. El LLMWrapper hace la llamada (con caché y fallback) y se devuelve.

Separar responsabilidades así permite testear cada pieza por su cuenta: los
tests de template no necesitan HTTP, y los del endpoint no necesitan un LLM.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from app.dependencies import get_llm_wrapper
from app.prompts.loader import (
    DEFAULT_VERSION,
    UnknownPromptVersionError,
    available_versions,
    render_estimation_prompt,
)
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    PromptPreview,
)
from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])

# NOTA: definir el Query una vez y reutilizarlo en los tres endpoints evita
# repetir el default y la descripción (que además aparece en Swagger /docs).
# La versión va como query param (?prompt_version=v2) y no en el body porque
# no es un dato del proyecto, sino una elección de "cómo" procesarlo.
PromptVersion = Query(
    default=DEFAULT_VERSION,
    description="Prompt template version to render (see GET /api/v1/prompts/versions)",
)


def _render(request: EstimationRequest, version: str) -> tuple[str, str]:
    """Render the prompt, turning an unknown version into a client error."""
    # NOTA: traducir excepciones de dominio a códigos HTTP es tarea del router.
    # El loader no sabe nada de HTTP: solo lanza UnknownPromptVersionError.
    # Aquí decidimos que eso es culpa del cliente → 422 Unprocessable Entity.
    try:
        return render_estimation_prompt(request, version=version)
    except UnknownPromptVersionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# Sync handler on purpose: FastAPI runs it in a worker thread, so the blocking
# LLM call does not stall the event loop.
# NOTA: si fuera ``async def`` y dentro hiciéramos una llamada bloqueante
# (wrapper.complete tarda segundos), congelaríamos el servidor entero mientras
# tanto: ninguna otra petición se atendería. Con ``def`` normal, FastAPI la
# ejecuta en un hilo aparte automáticamente.
@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(
    request: EstimationRequest,
    prompt_version: str = PromptVersion,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EstimationResponse:
    """Estimate a project from the typed form parameters."""
    # NOTA: ``request`` ya llega validado. Si el cliente mandó
    # "project_type": "MOBILE_APP" o una descripción de 5 letras, FastAPI ya
    # respondió 422 y esta función ni se ejecuta.
    # ``wrapper`` llega por inyección de dependencias (Depends): en producción
    # es el LLMWrapper real; en los tests se sustituye por un stub con
    # ``app.dependency_overrides`` sin tocar este código.
    system, user = _render(request, prompt_version)
    try:
        # NOTA: system y user viajan como DOS mensajes separados
        # ({"role": "system"} y {"role": "user"}), no concatenados en uno.
        result = wrapper.complete(system_prompt=system, user_message=user)
    except Exception as exc:  # noqa: BLE001 — every provider failure maps to 502
        log.error("estimation_endpoint_error", error=str(exc), error_type=type(exc).__name__)
        # NOTA: 502 Bad Gateway = "el servidor de más atrás falló". Distingue un
        # fallo del proveedor LLM (no es culpa del cliente → no es 4xx) de un
        # bug nuestro (500).
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc

    return EstimationResponse(
        text=result["estimation"],
        prompt_version=prompt_version,
        model=result["model"],
        provider=result["provider"],
        finish_reason=result["finish_reason"],
        usage=result["usage"],
        latency_ms=result["latency_ms"],
        cache_hit=result.get("cache_hit", False),
        cost_usd=result.get("cost_usd", 0.0),
    )


@router.post("/estimate/stream")
async def create_estimation_stream(
    request: EstimationRequest,
    prompt_version: str = PromptVersion,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EventSourceResponse:
    """Stream the same estimation as POST /estimate token by token via SSE.

    Emits ``token`` events, then one ``metrics`` event (JSON, including the
    ``prompt_version``) and a final ``done``.

    NOTA: Server-Sent Events (SSE) es un protocolo sencillo sobre HTTP: el
    servidor deja la conexión abierta y va enviando bloques de texto así::

        event: token
        data: Fase 1

        event: metrics
        data: {"model": "gpt-4o-mini", "usage": {...}, "prompt_version": "v1"}

        event: done
        data: [DONE]

    El cliente pinta cada ``token`` según llega, en vez de esperar a la
    respuesta completa.
    """
    system, user = _render(request, prompt_version)

    async def event_generator() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        metrics: dict = {}
        chunks = wrapper.complete_stream(
            system_prompt=system,
            user_message=user,
            metrics=metrics,
        )

        # NOTA: el iterador del wrapper es síncrono (bloquea mientras espera
        # al proveedor). Para no bloquear el event loop, cada next() se ejecuta
        # en un hilo con run_in_executor y lo esperamos con await.
        def _next_chunk() -> str | None:
            try:
                return next(chunks)
            except StopIteration:
                return None
            except Exception as exc:  # noqa: BLE001 — surface as SSE error event
                log.error("estimate_stream_failed", error=str(exc), error_type=type(exc).__name__)
                raise

        try:
            while True:
                chunk = await loop.run_in_executor(None, _next_chunk)
                if chunk is None:
                    break
                if chunk:
                    yield {"event": "token", "data": chunk}
            if metrics:
                yield {
                    "event": "metrics",
                    "data": json.dumps({**metrics, "prompt_version": prompt_version}),
                }
            yield {"event": "done", "data": "[DONE]"}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())


@router.post("/prompt/preview", response_model=PromptPreview)
async def preview_prompt(
    request: EstimationRequest,
    prompt_version: str = PromptVersion,
) -> PromptPreview:
    """Return the exact (system, user) messages a request renders to.

    No LLM call: lets clients (the Streamlit sidebar) show what is sent to the
    model and how it changes with each form parameter or prompt version.

    NOTA: endpoint didáctico y de depuración. Prueba en /docs a cambiar
    ``output_format`` o ``prompt_version`` y compara el ``system`` devuelto:
    verás exactamente qué bloque {% if %} del template se activa. Es gratis
    (no gasta tokens) porque nunca llama al modelo.
    """
    system, user = _render(request, prompt_version)
    return PromptPreview(system=system, user=user, prompt_version=prompt_version)


@router.get("/prompts/versions", response_model=list[str])
async def list_prompt_versions() -> list[str]:
    """List the available estimation prompt versions."""
    return available_versions()
