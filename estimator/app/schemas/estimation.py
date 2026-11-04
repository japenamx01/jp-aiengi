"""Typed contract between the client form and the estimation service.

The client no longer sends free text to be prompted around: it sends a bounded
set of typed parameters, and the service turns them into a prompt via the
versioned Jinja2 templates in :mod:`app.prompts`.

NOTA: ¿por qué un formulario tipado y no un textarea libre?
-----------------------------------------------------------
Con un chat, el usuario tiene que saber "hacer prompting": pedir el formato,
el nivel de detalle, el tipo de proyecto... y cada uno lo pide distinto. Como
aquí el espacio de tareas es acotado (siempre es "estima este proyecto"), es
mejor que el PRODUCTO decida cómo preguntar al modelo y que el usuario solo
aporte datos:

    Chat:        "hazme una estimación en tabla, breve, es una app móvil..."
    Formulario:  {"project_type": "mobile_app", "detail_level": "summary",
                  "output_format": "phases_table", "description": "..."}

Ventajas: validación automática (Pydantic rechaza valores inválidos con 422),
respuestas consistentes, y cada combinación se puede testear.
"""

from enum import Enum

from pydantic import BaseModel, Field

# NOTA: heredar de ``str`` y de ``Enum`` a la vez hace que cada miembro SEA un
# string: ProjectType.MOBILE_APP == "mobile_app" es True, y al serializar a
# JSON sale "mobile_app" (el VALOR, no el nombre "MOBILE_APP"). Por eso el
# cliente debe enviar los valores en minúscula exactamente como están aquí.


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class EstimationRequest(BaseModel):
    """Form submission: what to estimate and how the answer should look."""

    # NOTA: los límites de longitud protegen dos cosas: que haya algo que
    # estimar (mín. 20) y que nadie mande un texto enorme que dispare el coste
    # en tokens (máx. 2000). FastAPI los valida antes de ejecutar el endpoint.
    description: str = Field(min_length=20, max_length=2000)
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class EstimationResponse(BaseModel):
    """Free-text estimation plus the prompt version that produced it.

    The call metadata fields are additive (defaults keep the minimal contract
    ``{text, prompt_version}`` valid) and feed the client's metrics sidebar.
    """

    # NOTA: devolver ``prompt_version`` junto al texto es clave para la
    # trazabilidad: si un cliente se queja de una estimación, sabes con qué
    # versión del prompt se generó y puedes reproducirla.
    text: str
    prompt_version: str

    # NOTA: todo lo de abajo tiene valor por defecto, así que es "aditivo": un
    # cliente que solo lea {text, prompt_version} sigue funcionando.
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    # NOTA: default_factory llama a TokenUsage() para crear uno nuevo en cada
    # respuesta. Es la forma idiomática de dar un default "mutable" (un objeto,
    # una lista): en funciones o dataclasses, usar el mismo objeto como default
    # haría que todas las instancias lo compartieran (bug clásico de Python).
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    cache_hit: bool = False
    cost_usd: float = 0.0


class PromptPreview(BaseModel):
    """The exact (system, user) pair a request renders to, without calling the LLM."""

    system: str
    user: str
    prompt_version: str
