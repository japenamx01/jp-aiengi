"""Exact-match Redis cache for LLM responses.

The cache key is a SHA-256 of the *full* system prompt plus the user message
plus the generation knobs (model, max_tokens, thinking_budget). That means any
change in the form parameters or the prompt version implicitly invalidates the
cache without manual flushing, because those changes alter the rendered text.

NOTA: caché "exact-match"
-------------------------
Solo hay acierto si el prompt es IDÉNTICO carácter a carácter. "App de pádel"
y "app de padel" son claves distintas. Por eso es barata y segura (nunca
devuelve la respuesta de otra pregunta), pero acierta poco con texto libre. La
caché semántica (por significado) queda para una sesión posterior.

Como la clave incluye el system prompt renderizado, pasar de v1 a v2 o de
"summary" a "detailed" genera otra clave sola: nunca se sirve una respuesta
de una versión vieja del prompt.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis
import structlog

log = structlog.get_logger()


class EstimationCache:
    """Thin wrapper around redis-py with deterministic keying and TTL."""

    def __init__(self, redis_client: redis.Redis, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl

    @classmethod
    def from_url(cls, url: str, ttl: int = 86400) -> "EstimationCache":
        return cls(redis.from_url(url, decode_responses=True), ttl=ttl)

    # NOTA: json.dumps(sort_keys=True) da siempre el mismo string para el mismo
    # diccionario (sin importar el orden de las claves), y SHA-256 lo reduce a
    # 64 caracteres. Mismo input → misma clave, en cualquier máquina.
    @staticmethod
    def make_key(
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
        thinking_budget: int | None,
    ) -> str:
        payload = json.dumps(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "model": model,
                "max_tokens": max_tokens,
                "thinking_budget": thinking_budget,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"estimation:{digest}"

    def get(self, key: str) -> dict[str, Any] | None:
        # NOTA: si Redis está caído, se registra un warning y se devuelve None
        # (como un miss). La caché es una optimización: su fallo nunca debe
        # tumbar la estimación, solo hacerla más lenta y cara.
        try:
            cached = self.redis.get(key)
        except redis.RedisError as exc:
            log.warning("cache_get_failed", error=str(exc))
            return None
        if cached:
            log.info("cache_hit", key_prefix=key[:24])
            return json.loads(cached)
        log.info("cache_miss", key_prefix=key[:24])
        return None

    def set(self, key: str, response: dict[str, Any]) -> None:
        try:
            # NOTA: setex = SET + EXpire. La entrada se borra sola tras ttl s
            # (CACHE_TTL, por defecto 24 h), así Redis no crece sin límite.
            self.redis.setex(key, self.ttl, json.dumps(response))
            log.info("cache_stored", key_prefix=key[:24], ttl=self.ttl)
        except redis.RedisError as exc:
            log.warning("cache_set_failed", error=str(exc))
