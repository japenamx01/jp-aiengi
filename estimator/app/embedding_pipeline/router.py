"""HTTP layer of the embedding pipeline: ``POST /embeddings/ingest``.

NOTA: router "delgado": solo orquesta chunker -> embedder -> respuesta y traduce
fallos a códigos HTTP. La lógica vive en chunker.py y embedder.py, que se
prueban por separado sin FastAPI.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_chunker, get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimated_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger()

# NOTA: el enunciado fija el prefijo /embeddings (sin /api/v1), así que la ruta
# completa queda POST /embeddings/ingest.
router = APIRouter(prefix="/embeddings", tags=["embeddings"])


# NOTA: ``def`` y no ``async def``: el SDK de OpenAI que usamos es síncrono
# (bloquea mientras espera la respuesta). FastAPI ejecuta las funciones ``def`` en
# un pool de hilos, así una ingesta lenta no congela el resto de peticiones.
@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> IngestResponse:
    """Chunk the budgets (1 component = 1 chunk), embed every chunk and return
    the vectors plus aggregate stats. Nothing is persisted."""
    # NOTA: si el body no cumple el schema, FastAPI responde 422 antes de llegar
    # aquí; dentro del handler los datos ya están validados.
    if embedder is None:
        # NOTA: mensaje genérico al cliente y el motivo real en el log: no hace
        # falta contarle al exterior cómo está configurado el servidor.
        log.error("embeddings_ingest_failed", reason="embedder_unavailable_no_openai_key")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    chunks = chunker.chunk(request.budgets)
    log.info(
        "embeddings_ingest_received",
        total_budgets=len(request.budgets),
        total_chunks=len(chunks),
    )

    try:
        embedded = embedder.embed_many(chunks)
    except Exception as exc:  # noqa: BLE001 — cualquier fallo de la API de embeddings = 500
        log.error(
            "embeddings_ingest_failed",
            reason="embedding_api_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.") from exc

    total_tokens = sum(chunk.token_count for chunk in embedded)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd(total_tokens),
    )
    log.info("embeddings_ingest_done", model=embedder.model, **stats.model_dump())
    return IngestResponse(chunks=embedded, stats=stats)
