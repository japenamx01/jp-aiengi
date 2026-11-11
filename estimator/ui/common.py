"""Pieces shared by every Streamlit page: API location, labels and error helpers.

NOTA: cada página de ``ui/`` es un script independiente que Streamlit ejecuta
de arriba abajo; lo que comparten (URL de la API, etiquetas, cómo mostrar un
error) vive aquí para no copiarlo en cada una.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

# NOTA: el cliente importa las MISMAS clases Pydantic que valida el servicio.
# Un solo contrato compartido: si mañana cambia un enum, cliente y servicio se
# enteran a la vez, y los errores se detectan antes de salir del navegador.
from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000").rstrip("/")
ESTIMATE_ENDPOINT = f"{API_BASE_URL}/api/v1/estimate"
STREAM_ENDPOINT = f"{API_BASE_URL}/api/v1/estimate/stream"
PREVIEW_ENDPOINT = f"{API_BASE_URL}/api/v1/prompt/preview"
VERSIONS_ENDPOINT = f"{API_BASE_URL}/api/v1/prompts/versions"
SESSIONS_ENDPOINT = f"{API_BASE_URL}/api/v1/sessions"

# NOTA: el usuario ve textos amigables ("Phases table") pero el widget
# devuelve el enum (OutputFormat.PHASES_TABLE). format_func=LABELS.get hace la
# traducción solo para mostrar; el valor real nunca cambia.
LABELS: dict[Any, str] = {
    ProjectType.MOBILE_APP: "Mobile app",
    ProjectType.WEB_SAAS: "Web SaaS",
    ProjectType.INTERNAL_TOOL: "Internal tool",
    ProjectType.DATA_PIPELINE: "Data pipeline",
    DetailLevel.SUMMARY: "Summary",
    DetailLevel.MEDIUM: "Medium",
    DetailLevel.DETAILED: "Detailed",
    OutputFormat.PHASES_TABLE: "Phases table",
    OutputFormat.LINE_ITEMS: "Line items",
    OutputFormat.NARRATIVE: "Narrative",
}


# NOTA: como el script se re-ejecuta en cada interacción, sin caché haríamos
# una petición HTTP a /prompts/versions cada vez. Con ttl=60 se pide como mucho
# una vez por minuto; si añades v3 en el servidor, aparece en <= 60 s.
@st.cache_data(ttl=60, show_spinner=False)
def fetch_versions() -> list[str]:
    """Prompt versions offered by the API; falls back to v1 if it is down."""
    try:
        response = httpx.get(VERSIONS_ENDPOINT, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return ["v1"]
    return response.json()


def error_detail(exc: httpx.HTTPError) -> str:
    """Prefer the API's own error message over httpx's generic one."""
    # NOTA: httpx diría "Client error '422 Unprocessable Entity'..." (poco útil).
    # FastAPI manda el motivo real en el campo "detail" del JSON; lo mostramos.
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return str(exc.response.json().get("detail", exc))
        except ValueError:
            pass
    return str(exc)


def first_error(exc: ValidationError) -> str:
    # NOTA: exc.errors() es una lista de dicts como
    # {"loc": ("description",), "msg": "String should have at least 20 characters"}.
    # Mostramos el primero en formato "campo: motivo".
    err = exc.errors()[0]
    return f"{err['loc'][0]}: {err['msg']}"


def render_service_info(endpoint: str) -> None:
    """Sidebar block with the endpoint in use and the configured models."""
    st.header("Service")
    st.code(endpoint, language="text")
    primary = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{primary}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")


def render_call_metrics(last: dict) -> None:
    """Model, prompt version, tokens, response time and cache flag of a call."""
    usage = last.get("usage", {})
    # Plain text: model ids are too long for st.metric's large font.
    st.markdown(f"**Model:** `{last.get('model') or 'unknown'}`")
    st.markdown(f"**Prompt version:** `{last['prompt_version']}`")
    tokens_in, tokens_out = st.columns(2)
    tokens_in.metric("Input tokens", f"{usage.get('input_tokens', 0):,}", border=True)
    tokens_out.metric("Output tokens", f"{usage.get('output_tokens', 0):,}", border=True)
    st.metric(
        "Response time",
        f"{last['elapsed_s']:.2f} s",
        help="Measured in the browser session, from request to the end of the answer.",
        border=True,
    )
    if last.get("cache_hit"):
        st.badge("Served from Redis cache", icon=":material/bolt:", color="green")
