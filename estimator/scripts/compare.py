#!/usr/bin/env python3
"""Embedding sanity check: cosine similarity between two texts.

NOTA: embebe dos textos con el MISMO ``OpenAIEmbedder`` que usa el endpoint y
calcula la similitud coseno a mano, solo con la biblioteca estándar (sin numpy
ni scikit-learn).

Usage::

    # Fuera del contenedor (desde estimator/, con .env):
    uv run python scripts/compare.py \\
        --text-a "OAuth 2.0 authentication backend for fintech" \\
        --text-b "JWT-based authorization service for banking app"

    # Dentro del contenedor:
    docker compose exec estimator python scripts/compare.py --text-a "..." --text-b "..."
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# NOTA: al ejecutar ``python scripts/compare.py`` Python añade scripts/ a
# sys.path, no la raíz del proyecto, y ``import app`` fallaría. Añadimos la raíz
# (estimator/ en local, /app en el contenedor).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos(θ) = (a · b) / (‖a‖ · ‖b‖).

    NOTA: mide el ÁNGULO entre los vectores, no su longitud:
       1.0 -> misma dirección (mismo significado)
       0.0 -> perpendiculares (nada en común)
      -1.0 -> opuestos (en la práctica, los embeddings de texto casi nunca bajan de 0)
    Los vectores de OpenAI ya vienen normalizados (‖v‖ = 1), así que aquí el
    coseno coincide con el producto escalar; dividir por las normas lo hace
    correcto para cualquier vector.
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must have the same length ({len(a)} != {len(b)})")
    dot = sum(x * y for x, y in zip(a, b))  # producto escalar
    norm_a = math.sqrt(sum(x * x for x in a))  # longitud (norma L2) de a
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0  # un vector nulo no tiene dirección: no hay ángulo que medir
    return dot / (norm_a * norm_b)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cosine similarity between two embedded texts.")
    parser.add_argument("--text-a", required=True, help="First text.")
    parser.add_argument("--text-b", required=True, help="Second text.")
    args = parser.parse_args()

    # NOTA: import tardío: así ``cosine_similarity`` se puede importar en los
    # tests sin necesitar configuración ni claves.
    from app.dependencies import build_embedder

    embedder = build_embedder()
    if embedder is None:
        print("ERROR: OPENAI_API_KEY is not set (check your .env).", file=sys.stderr)
        return 1

    vec_a = embedder.embed_one(args.text_a)
    vec_b = embedder.embed_one(args.text_b)

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Model: {embedder.model} ({len(vec_a)} dims)")
    print(f"Cosine similarity: {cosine_similarity(vec_a, vec_b):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
