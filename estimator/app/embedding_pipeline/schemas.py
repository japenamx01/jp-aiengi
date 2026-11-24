"""Pydantic models for the embedding pipeline.

NOTA: el lado de entrada refleja el JSON normalizado de presupuestos históricos
(un presupuesto con su lista de componentes). El lado de salida lleva los chunks
listos para embeber y, una vez embebidos, sus vectores y las estadísticas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# NOTA: universo cerrado de valores del dataset. Con un Literal, un sector mal
# escrito ("fintech", "Finance") falla con 422 en la puerta en vez de colarse en
# la metadata, donde en la Sesión 08 rompería los filtros sin hacer ruido.
Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


# --------------------------------------------------------------------------- #
# Entrada: presupuestos históricos
# --------------------------------------------------------------------------- #


class ClientMetadata(BaseModel):
    """Who the budget belongs to."""

    name: str = Field(min_length=1, description="Client company name.")
    sector: Sector = Field(description="Client business sector.")
    country: str = Field(min_length=2, description="Country code, e.g. 'ES'.")


class BudgetComponent(BaseModel):
    """One line item of a historical budget. It becomes exactly one chunk."""

    component_id: str = Field(min_length=1, description="Id within the budget, e.g. 'AUTH-001'.")
    name: str = Field(min_length=1, description="Short component name.")
    description: str = Field(min_length=1, description="Detailed description of the work.")
    tech_stack: list[str] = Field(default_factory=list, description="Technologies involved.")
    estimated_hours: int = Field(ge=0, description="Hours estimated for this component.")
    complexity: Complexity = Field(description="Coarse complexity bucket.")
    dependencies: list[str] = Field(
        default_factory=list, description="component_ids this one depends on."
    )


class Budget(BaseModel):
    """A complete historical budget with its components."""

    budget_id: str = Field(min_length=1, description="Budget id, e.g. 'BUD-2024-014'.")
    client_metadata: ClientMetadata
    project_summary: str = Field(min_length=1, description="One-line project summary.")
    main_technology: str = Field(min_length=1, description="Primary stack of the project.")
    year: int = Field(ge=2000, le=2100, description="Year the budget was produced.")
    total_estimated_hours: int = Field(ge=0, description="Sum of component hours, as recorded.")
    # NOTA: un presupuesto sin componentes no produciría ningún chunk; mejor
    # rechazarlo con 422 que devolver un 200 vacío que parezca un éxito.
    components: list[BudgetComponent] = Field(min_length=1, description="Budget line items.")


# --------------------------------------------------------------------------- #
# Salida: chunks y vectores
# --------------------------------------------------------------------------- #


class Chunk(BaseModel):
    """A fragment ready to be embedded.

    NOTA: la separación clave del ejercicio:
      - ``text``     -> lo ÚNICO que se envía al modelo y se convierte en vector.
      - ``metadata`` -> viaja junto al chunk pero NO se embebe; son campos exactos
                        para filtrar (sector = finance, year >= 2024) cuando haya
                        base vectorial. Filtrar por igualdad es trabajo de SQL, no
                        de la similitud semántica.
    """

    chunk_id: str = Field(description="Traceable id: '{budget_id}::{component_id}'.")
    text: str = Field(description="Embeddable text: parent header + component detail.")
    metadata: dict = Field(default_factory=dict, description="Filterable, non-embedded fields.")
    token_count: int = Field(ge=0, description="Tokens of ``text`` according to tiktoken.")


class EmbeddedChunk(Chunk):
    """A :class:`Chunk` with its vector attached (herencia: mismo chunk + embedding)."""

    embedding: list[float] = Field(
        description="Dense vector (1536 floats for text-embedding-3-small)."
    )


class IngestRequest(BaseModel):
    """Body of ``POST /embeddings/ingest``."""

    budgets: list[Budget] = Field(min_length=1, description="Budgets to chunk and embed.")


class IngestStats(BaseModel):
    """Aggregate counters of one ingest call.

    NOTA: el enunciado pide ``stats`` como dict; un modelo produce el mismo JSON
    pero queda tipado y documentado en Swagger.
    """

    total_budgets: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)


class IngestResponse(BaseModel):
    """Response of ``POST /embeddings/ingest``."""

    chunks: list[EmbeddedChunk]
    stats: IngestStats
