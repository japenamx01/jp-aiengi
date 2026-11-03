"""Render versioned Jinja2 prompt templates into (system, user) message pairs.

Layout: ``app/prompts/<task>/<version>/{system,user,examples}.j2``. A published
version is immutable: changing a prompt means adding ``v2/`` next to ``v1/``,
never editing ``v1/`` in place, so every logged ``prompt_version`` keeps
pointing at the exact text that produced a response.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import EstimationRequest

log = structlog.get_logger()

PROMPTS_DIR = Path(__file__).resolve().parent
ESTIMATION_DIR = PROMPTS_DIR / "estimation"
DEFAULT_VERSION = "v1"

_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    # Fail loudly on a missing/typo'd variable instead of rendering "".
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    autoescape=False,
)


class UnknownPromptVersionError(ValueError):
    """Raised when a request asks for a prompt version that does not exist."""


def available_versions() -> list[str]:
    """Return the estimation prompt versions found on disk, sorted (v1, v2, ...)."""
    return sorted(
        p.name for p in ESTIMATION_DIR.iterdir() if p.is_dir() and (p / "system.j2").is_file()
    )


def render_estimation_prompt(
    request: EstimationRequest, version: str = DEFAULT_VERSION
) -> tuple[str, str]:
    """Render the estimation prompt for ``request`` and return ``(system, user)``."""
    # Whitelisting against the directories on disk also rules out path traversal.
    if version not in available_versions():
        raise UnknownPromptVersionError(
            f"Unknown prompt version '{version}'. Available: {', '.join(available_versions())}"
        )

    # mode="json" turns the enums into their string values ("phases_table"),
    # which is what the templates compare against.
    context = request.model_dump(mode="json")
    system = _env.get_template(f"estimation/{version}/system.j2").render(**context).strip()
    user = _env.get_template(f"estimation/{version}/user.j2").render(**context).strip()

    log.info(
        "prompt_rendered",
        prompt="estimation",
        prompt_version=version,
        prompt_hash=hashlib.sha256(f"{system}\n\n{user}".encode()).hexdigest()[:12],
        system_chars=len(system),
        user_chars=len(user),
        project_type=context["project_type"],
        detail_level=context["detail_level"],
        output_format=context["output_format"],
    )
    return system, user
