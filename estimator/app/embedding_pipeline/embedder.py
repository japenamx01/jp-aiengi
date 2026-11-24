"""OpenAI embedder (``text-embedding-3-small``).

NOTA: qué es un embedding: el modelo convierte un texto en un vector de 1536
floats. Textos con significado parecido quedan cerca en ese espacio (coseno
alto) aunque no compartan palabras. Este módulo solo pide esos vectores; no
los guarda ni busca en ellos.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import structlog
from openai import RateLimitError

from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger()

# NOTA: dimensión por defecto del modelo. No la recortamos (el parámetro
# ``dimensions`` permite pedir menos gracias a Matryoshka, pero eso se discute en
# el directo).
DEFAULT_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# NOTA: el endpoint acepta una LISTA de textos por llamada. Con lotes de 100,
# los 52 chunks del sample salen en 1 petición HTTP en vez de 52 seguidas:
# menos latencia total y menos riesgo de rate limit.
BATCH_SIZE = 100

# NOTA: precio orientativo que CAMBIA con el tiempo: revísalo en la página de
# precios de OpenAI. text-embedding-3-small: 0.02 USD por 1M tokens de entrada
# (los embeddings no tienen tokens de salida).
PRICE_PER_MILLION_TOKENS_USD = 0.02

# NOTA: backoff exponencial: 3 reintentos esperando 1s, 2s y 4s. Solo para
# RateLimitError (429), que se resuelve esperando. Una clave inválida (401) o
# una entrada demasiado larga (400) fallarían igual al reintentar, así que esos
# errores se propagan enseguida.
RETRY_BACKOFF_SECONDS = (1, 2, 4)


def estimated_cost_usd(total_tokens: int) -> float:
    """Cost in USD of embedding ``total_tokens`` input tokens."""
    return total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD


class OpenAIEmbedder:
    """Thin wrapper over ``client.embeddings.create`` with batching and retries.

    NOTA: recibe el cliente ya construido (inyección de dependencias). En
    producción es ``openai.OpenAI``; en los tests, un objeto falso con el mismo
    método ``embeddings.create``, así se prueba la lógica sin red ni tokens.
    ``sleep`` también se inyecta para que los tests de reintento no esperen 7 s.
    """

    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_MODEL,
        batch_size: int = BATCH_SIZE,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._model

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text (used by ``scripts/compare.py``)."""
        return self._create([text])[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed every chunk, in order, one API call per batch."""
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]

            t0 = time.perf_counter()
            vectors = self._create([chunk.text for chunk in batch])
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            log.info(
                "embedding_batch_done",
                batch_index=start // self._batch_size,
                chunks=len(batch),
                tokens=sum(chunk.token_count for chunk in batch),
                latency_ms=latency_ms,
                model=self._model,
            )

            # NOTA: zip empareja cada chunk con SU vector; funciona porque
            # _create devuelve los vectores en el mismo orden que los textos.
            for chunk, vector in zip(batch, vectors, strict=True):
                embedded.append(EmbeddedChunk(**chunk.model_dump(), embedding=vector))
        return embedded

    def _create(self, texts: list[str]) -> list[list[float]]:
        """One embeddings call with exponential backoff on rate limits only."""
        attempts = (0, *RETRY_BACKOFF_SECONDS)  # 0 = primer intento, sin espera
        for attempt, wait in enumerate(attempts):
            if wait:
                self._sleep(wait)
            try:
                response = self._client.embeddings.create(model=self._model, input=texts)
            except RateLimitError:
                if attempt == len(attempts) - 1:
                    log.error("embedding_rate_limit_exhausted", attempts=len(attempts))
                    raise
                log.warning("embedding_rate_limited", retry_in_s=attempts[attempt + 1])
                continue
            # NOTA: cada item trae ``index`` (su posición en la entrada).
            # Ordenamos por él para no depender de que la API respete el orden.
            data = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in data]
        raise AssertionError("unreachable")  # el bucle siempre devuelve o relanza
