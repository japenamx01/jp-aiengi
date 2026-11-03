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

PromptVersion = Query(
    default=DEFAULT_VERSION,
    description="Prompt template version to render (see GET /api/v1/prompts/versions)",
)


def _render(request: EstimationRequest, version: str) -> tuple[str, str]:
    """Render the prompt, turning an unknown version into a client error."""
    try:
        return render_estimation_prompt(request, version=version)
    except UnknownPromptVersionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# Sync handler on purpose: FastAPI runs it in a worker thread, so the blocking
# LLM call does not stall the event loop.
@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(
    request: EstimationRequest,
    prompt_version: str = PromptVersion,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EstimationResponse:
    """Estimate a project from the typed form parameters."""
    system, user = _render(request, prompt_version)
    try:
        result = wrapper.complete(system_prompt=system, user_message=user)
    except Exception as exc:  # noqa: BLE001 — every provider failure maps to 502
        log.error("estimation_endpoint_error", error=str(exc), error_type=type(exc).__name__)
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
    """
    system, user = _render(request, prompt_version)
    return PromptPreview(system=system, user=user, prompt_version=prompt_version)


@router.get("/prompts/versions", response_model=list[str])
async def list_prompt_versions() -> list[str]:
    """List the available estimation prompt versions."""
    return available_versions()
