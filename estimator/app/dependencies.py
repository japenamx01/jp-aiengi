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

from app.config import get_settings
from app.services.cache import EstimationCache
from app.services.llm_wrapper import LLMWrapper


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
