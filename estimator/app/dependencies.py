"""FastAPI dependency factories for shared singletons (cache and LLM wrapper).

NOTA: @lru_cache sin argumentos convierte la función en un "singleton": la
primera llamada crea el objeto y las siguientes devuelven EL MISMO. Así hay
una sola conexión a Redis y un solo Router de LiteLLM para todo el proceso, en
vez de crear uno por petición. Los endpoints lo piden con
``Depends(get_llm_wrapper)`` y los tests lo sustituyen con
``app.dependency_overrides``.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.config import get_settings
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.services.cache import EstimationCache
from app.services.llm_wrapper import LLMWrapper
from app.services.sessions import SessionStore


@lru_cache
def get_cache() -> EstimationCache:
    settings = get_settings()
    return EstimationCache.from_url(settings.REDIS_URL, ttl=settings.CACHE_TTL)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    return LLMWrapper(
        openai_api_key=settings.OPENAI_API_KEY,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        timeout=settings.LLM_TIMEOUT,
        num_retries=settings.LLM_RETRIES,
        cache=get_cache(),
    )


@lru_cache
def get_session_store() -> SessionStore:
    # NOTA: el mismo truco de singleton sirve aquí para algo más importante:
    # TODAS las peticiones deben ver el MISMO diccionario de sesiones. Si se
    # creara un SessionStore por petición, cada turno empezaría sin memoria.
    return SessionStore(max_turns=get_settings().MAX_TURNS)


# --- Session 7: embedding pipeline ------------------------------------------


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    # NOTA: el chunker no tiene estado; el singleton solo evita crear un objeto
    # por petición y deja un punto único para sustituirlo en los tests.
    return JSONStructuralChunker()


def build_embedder() -> OpenAIEmbedder | None:
    """Build an embedder from settings, or ``None`` if OPENAI_API_KEY is missing.

    NOTA: compartido por el endpoint y por scripts/compare.py, así los dos usan
    exactamente el mismo modelo y la misma configuración.
    """
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        return None
    # NOTA: max_retries=0 apaga los reintentos internos del SDK (2 por defecto).
    # Si no, se sumarían a nuestro backoff 1s/2s/4s y no sabríamos cuántas
    # llamadas reales se hicieron. Los reintentos quedan en un solo sitio.
    client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.LLM_TIMEOUT, max_retries=0)
    return OpenAIEmbedder(client=client, model=settings.EMBEDDING_MODEL)


@lru_cache
def get_embedder() -> OpenAIEmbedder | None:
    # NOTA: devuelve None en vez de lanzar una excepción: la app arranca aunque
    # solo haya clave de Anthropic, y el endpoint traduce ese None a un 500.
    return build_embedder()
