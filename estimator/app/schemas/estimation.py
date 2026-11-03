"""Typed contract between the client form and the estimation service.

The client no longer sends free text to be prompted around: it sends a bounded
set of typed parameters, and the service turns them into a prompt via the
versioned Jinja2 templates in :mod:`app.prompts`.
"""

from enum import Enum

from pydantic import BaseModel, Field


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

    text: str
    prompt_version: str

    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    cache_hit: bool = False
    cost_usd: float = 0.0


class PromptPreview(BaseModel):
    """The exact (system, user) pair a request renders to, without calling the LLM."""

    system: str
    user: str
    prompt_version: str
