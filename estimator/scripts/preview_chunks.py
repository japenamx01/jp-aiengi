#!/usr/bin/env python3
"""Preview the chunks of a budgets file WITHOUT calling OpenAI (cost 0).

NOTA: antes de pagar por embeber, mira qué vas a embeber. Este script valida el
JSON, lo pasa por el chunker y muestra cuántos chunks salen, cuántos tokens
tienen, el más grande y el texto exacto de uno de ellos.

Usage::

    uv run python scripts/preview_chunks.py                 # data/budgets_sample.json
    uv run python scripts/preview_chunks.py --show 3        # print 3 chunk texts
    docker compose exec estimator python scripts/preview_chunks.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import DEFAULT_SAMPLE  # noqa: F401  (also fixes sys.path)

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import estimated_cost_usd
from app.embedding_pipeline.schemas import Budget


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview structural chunks (no API calls).")
    parser.add_argument("--file", type=Path, default=DEFAULT_SAMPLE, help="Budgets JSON array.")
    parser.add_argument("--show", type=int, default=1, help="How many chunk texts to print.")
    args = parser.parse_args()

    budgets = [Budget.model_validate(item) for item in json.loads(args.file.read_text())]
    chunks = JSONStructuralChunker().chunk(budgets)

    tokens = [c.token_count for c in chunks]
    biggest = max(chunks, key=lambda c: c.token_count)
    print(f"Budgets: {len(budgets)}  ->  chunks: {len(chunks)}")
    print(
        f"Tokens per chunk: min {min(tokens)} / avg {sum(tokens) / len(tokens):.0f} / "
        f"max {max(tokens)} ({biggest.chunk_id})"
    )
    print(f"Total tokens: {sum(tokens)}  ->  estimated cost: ${estimated_cost_usd(sum(tokens)):.6f}")

    for chunk in chunks[: args.show]:
        print(f"\n--- {chunk.chunk_id} ({chunk.token_count} tokens) ---")
        print(chunk.text)
        print(f"metadata (not embedded): {chunk.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
