"""Test doubles for the embedding pipeline (no network, no tokens).

NOTA: imitan solo la parte de ``openai.OpenAI`` que usa el embedder:
``client.embeddings.create(model=..., input=[...])`` devolviendo un objeto con
``.data[i].embedding`` y ``.data[i].index``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import RateLimitError

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "budgets_sample.json"


def load_sample() -> list[dict]:
    return json.loads(SAMPLE_PATH.read_text())


def rate_limit_error() -> RateLimitError:
    # NOTA: las excepciones del SDK necesitan una respuesta HTTP real para
    # construirse; con httpx se fabrica un 429 sin tocar la red.
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    return RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )


class FakeEmbeddings:
    """Records every call and returns deterministic vectors.

    El vector de cada texto es ``[len(text), position_in_batch, 0.0]``, así un
    test puede comprobar que cada chunk recibió SU vector y no el de otro.
    ``failures`` es una lista de excepciones a lanzar en las primeras llamadas.
    """

    def __init__(self, failures: list[Exception] | None = None, reverse: bool = False) -> None:
        self.calls: list[list[str]] = []
        self._failures = list(failures or [])
        self._reverse = reverse

    def create(self, *, model: str, input: list[str]):  # noqa: A002 (SDK signature)
        self.calls.append(list(input))
        if self._failures:
            raise self._failures.pop(0)
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text)), float(i), 0.0])
            for i, text in enumerate(input)
        ]
        if self._reverse:  # simula una API que devuelve los items desordenados
            data.reverse()
        return SimpleNamespace(data=data)


class FakeOpenAI:
    def __init__(self, **kwargs) -> None:
        self.embeddings = FakeEmbeddings(**kwargs)
