"""Structural chunker for budget JSON.

NOTA: estrategia "estructural": confiamos en la forma del documento.
Un ``BudgetComponent`` = exactamente un ``Chunk``.

- Sin solapamiento y sin cortar por tamaño fijo: un componente es una unidad de
  significado completa y cabe de sobra en el modelo (límite de 8191 tokens).
  Si alguna descripción sale enorme, ``token_count`` lo deja a la vista; es un
  dato para discutir, no algo que esconder partiéndolo.
- Cabecera de contexto (contextual chunk header): a cada chunk se le antepone el
  contexto del presupuesto padre. Sin ella, "Authentication backend" de un banco
  y de una tienda online darían casi el mismo vector.
"""

from __future__ import annotations

import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk

# NOTA: contamos con el MISMO tokenizador que usa el modelo de embeddings
# (cl100k_base), así token_count coincide con lo que OpenAI factura. Cargar el
# encoding es caro, por eso se resuelve una sola vez al importar el módulo y no
# en cada chunk.
TOKENIZER_MODEL = "text-embedding-3-small"
_ENCODING = tiktoken.encoding_for_model(TOKENIZER_MODEL)


def count_tokens(text: str) -> int:
    """Number of tokens the embedding model will see for ``text``."""
    return len(_ENCODING.encode(text))


class JSONStructuralChunker:
    """Turns budgets into one chunk per component.

    NOTA: no tiene estado ni dependencias externas: es una función pura
    (mismos presupuestos -> mismos chunks). Por eso se prueba sin red ni mocks.
    """

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                chunks.append(self._chunk_component(budget, component))
        return chunks

    def _chunk_component(self, budget: Budget, component: BudgetComponent) -> Chunk:
        text = render_chunk_text(budget, component)
        return Chunk(
            # NOTA: id trazable: dado un vector, sabes de qué presupuesto y
            # componente salió (BUD-2024-014::AUTH-001).
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            # NOTA: campos exactos para filtrar más adelante. No se embeben: el
            # vector capta el significado y estos campos sirven para el WHERE de SQL.
            metadata={
                "budget_id": budget.budget_id,
                "component_id": component.component_id,
                "client_sector": budget.client_metadata.sector,
                "main_technology": budget.main_technology,
                "year": budget.year,
                "complexity": component.complexity,
                "estimated_hours": component.estimated_hours,
            },
            token_count=count_tokens(text),
        )


def render_chunk_text(budget: Budget, component: BudgetComponent, *, header: bool = True) -> str:
    """Readable text that gets embedded: parent header + component detail.

    NOTA: ``header=False`` existe solo para el experimento didáctico de
    ``scripts/header_effect.py`` (medir cuánto aporta la cabecera). El pipeline
    siempre usa la cabecera.
    """
    body = (
        f"Component: {component.name}\n"
        f"Description: {component.description}\n"
        f"Tech stack: {', '.join(component.tech_stack)}\n"
        f"Complexity: {component.complexity}\n"
        f"Estimated hours: {component.estimated_hours}"
    )
    if not header:
        return body
    return (
        f"[Project: {budget.project_summary}]\n"
        f"[Client sector: {budget.client_metadata.sector} | "
        f"Year: {budget.year} | Main tech: {budget.main_technology}]\n"
        f"\n" + body
    )
