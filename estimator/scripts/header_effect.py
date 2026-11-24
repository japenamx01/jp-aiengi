#!/usr/bin/env python3
"""Didactic experiment: how much does the contextual chunk header help?

NOTA: el sample tiene componentes casi iguales en sectores distintos (un
"inventario" en e-commerce y otro en una farmacia hospitalaria). Para cada caso
embebemos una consulta, el componente correcto (target) y el señuelo (decoy) de
otro sector, CON y SIN la cabecera del presupuesto padre, y comparamos cosenos.

Si la cabecera sirve, la ventaja del target sobre el decoy (gap) crece al
añadirla. Es una medida por parejas, igual que compare.py; NO es búsqueda
semántica (eso llega en la Sesión 08). Coste: ~800 tokens (< 0.0001 USD).

Usage::

    uv run python scripts/header_effect.py
    docker compose exec estimator python scripts/header_effect.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dependencies import build_embedder  # noqa: E402
from app.embedding_pipeline.chunker import count_tokens, render_chunk_text  # noqa: E402
from app.embedding_pipeline.embedder import estimated_cost_usd  # noqa: E402
from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk  # noqa: E402
from scripts.compare import cosine_similarity  # noqa: E402

SAMPLE = ROOT / "data" / "budgets_sample.json"

# (consulta, chunk_id correcto, chunk_id señuelo de otro sector)
CASES = [
    (
        "Inventory management for a hospital pharmacy",
        "BUD-2024-011::STOCK-002",  # healthcare: Inventory and reorder
        "BUD-2024-007::INV-003",  # ecommerce: Real-time inventory sync
    ),
    (
        "Sensor data ingestion for a factory production line",
        "BUD-2024-013::TELE-001",  # industrial: Sensor telemetry ingestion
        "BUD-2024-010::INGEST-001",  # healthcare: Wearable data ingestion
    ),
    (
        "Onboarding and risk checks for new clients of an investment firm",
        "BUD-2024-004::ONB-003",  # finance: Client onboarding and risk profiling
        "BUD-2024-006::VENDOR-001",  # ecommerce: Vendor onboarding and KYC
    ),
]


def main() -> int:
    embedder = build_embedder()
    if embedder is None:
        print("ERROR: OPENAI_API_KEY is not set (check your .env).", file=sys.stderr)
        return 1

    components: dict[str, tuple[Budget, BudgetComponent]] = {}
    for item in json.loads(SAMPLE.read_text()):
        budget = Budget.model_validate(item)
        for comp in budget.components:
            components[f"{budget.budget_id}::{comp.component_id}"] = (budget, comp)

    # NOTA: juntamos TODOS los textos (consultas + componentes en sus dos
    # versiones) en una sola lista para pedirlos en un único lote.
    texts: dict[str, str] = {}
    for query, target, decoy in CASES:
        texts[f"q::{query}"] = query
        for cid in (target, decoy):
            budget, comp = components[cid]
            texts[f"with::{cid}"] = render_chunk_text(budget, comp, header=True)
            texts[f"without::{cid}"] = render_chunk_text(budget, comp, header=False)

    batch = [Chunk(chunk_id=k, text=t, token_count=count_tokens(t)) for k, t in texts.items()]
    vectors = {c.chunk_id: c.embedding for c in embedder.embed_many(batch)}
    tokens = sum(c.token_count for c in batch)

    print(f"Model: {embedder.model} | {len(batch)} texts | {tokens} tokens | "
          f"${estimated_cost_usd(tokens):.6f}\n")
    print(f"{'header':<8} {'target':>7} {'decoy':>7} {'gap':>7}")
    for query, target, decoy in CASES:
        print(f"\nQuery: {query!r}\n  target={target}  decoy={decoy}")
        q = vectors[f"q::{query}"]
        for mode in ("without", "with"):
            t = cosine_similarity(q, vectors[f"{mode}::{target}"])
            d = cosine_similarity(q, vectors[f"{mode}::{decoy}"])
            print(f"  {mode:<8} {t:7.4f} {d:7.4f} {t - d:+7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
