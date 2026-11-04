"""Streamlit form UI for the estimator.

The client no longer sends free text to a chat: a form produces a typed
``EstimationRequest`` (the same Pydantic model the service validates against)
and POSTs it to ``/api/v1/estimate``. The endpoint URL is read from
``ESTIMATOR_API_BASE_URL`` (loaded from the same ``.env`` as the API), so the
same UI works against a local uvicorn or against docker-compose.

The sidebar shows the metrics of the last call and the exact system/user
messages it rendered to, fetched from ``POST /api/v1/prompt/preview``.

NOTA: cómo "piensa" Streamlit
-----------------------------
Streamlit vuelve a ejecutar ESTE SCRIPT ENTERO de arriba abajo cada vez que el
usuario interactúa (pulsa un botón, cambia un widget...). No hay callbacks ni
rutas como en una web clásica: el script se re-ejecuta y redibuja la página.
Consecuencias que verás en el código:
  - Lo que deba sobrevivir entre ejecuciones se guarda en ``st.session_state``.
  - Lo caro de calcular se cachea con ``@st.cache_data``.
  - Dentro de un ``st.form`` los widgets NO provocan re-ejecuciones al cambiar;
    solo el botón de enviar lo hace. Así no se llama a la API a cada tecla.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

# NOTA: el cliente importa las MISMAS clases Pydantic que valida el servicio.
# Un solo contrato compartido: si mañana cambia un enum, cliente y servicio se
# enteran a la vez, y los errores se detectan antes de salir del navegador.
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000").rstrip("/")
ESTIMATE_ENDPOINT = f"{API_BASE_URL}/api/v1/estimate"
PREVIEW_ENDPOINT = f"{API_BASE_URL}/api/v1/prompt/preview"
VERSIONS_ENDPOINT = f"{API_BASE_URL}/api/v1/prompts/versions"

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

st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption(
    "Describe the project and choose how you want the estimation. The service turns "
    "these parameters into a versioned Jinja2 prompt."
)


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


def _error_detail(exc: httpx.HTTPError) -> str:
    """Prefer the API's own error message over httpx's generic one."""
    # NOTA: httpx diría "Client error '422 Unprocessable Entity'..." (poco útil).
    # FastAPI manda el motivo real en el campo "detail" del JSON; lo mostramos.
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return str(exc.response.json().get("detail", exc))
        except ValueError:
            pass
    return str(exc)


def _first_error(exc: ValidationError) -> str:
    # NOTA: exc.errors() es una lista de dicts como
    # {"loc": ("description",), "msg": "String should have at least 20 characters"}.
    # Mostramos el primero en formato "campo: motivo".
    err = exc.errors()[0]
    return f"{err['loc'][0]}: {err['msg']}"


# NOTA: inicialización de session_state en un único sitio. Solo se ejecuta la
# primera vez (cuando la clave aún no existe); en las re-ejecuciones siguientes
# se conservan los valores guardados de la llamada anterior.
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.last_metrics = None
    st.session_state.last_prompt = None

versions = fetch_versions()

# NOTA: cada widget del formulario corresponde a un campo de EstimationRequest.
# segmented_control con required=True obliga a tener siempre una opción
# elegida (sin required, el usuario podría desmarcarla y devolvería None).
with st.form("estimation_form"):
    description = st.text_area(
        "Project description",
        height=160,
        max_chars=2000,
        placeholder="E.g. A booking app for padel courts with online payments and a club admin panel.",
        help="Between 20 and 2000 characters.",
    )
    project_type = st.selectbox("Project type", list(ProjectType), format_func=LABELS.get)
    detail_level = st.segmented_control(
        "Level of detail",
        list(DetailLevel),
        default=DetailLevel.MEDIUM,
        required=True,
        format_func=LABELS.get,
    )
    output_format = st.segmented_control(
        "Output format",
        list(OutputFormat),
        default=OutputFormat.PHASES_TABLE,
        required=True,
        format_func=LABELS.get,
    )
    prompt_version = st.segmented_control(
        "Prompt version",
        versions,
        default=versions[0],
        required=True,
        help="v2 uses a client-facing tone with an executive summary.",
    )
    submitted = st.form_submit_button("Estimate", type="primary", icon=":material/calculate:")

# NOTA: ``submitted`` solo es True en la re-ejecución causada por el botón.
if submitted:
    # NOTA: validación en el cliente con la misma clase Pydantic. Si la
    # descripción es corta, se avisa aquí sin llegar a llamar a la API.
    try:
        request = EstimationRequest(
            description=description.strip(),
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )
    except ValidationError as exc:
        st.warning(_first_error(exc), title="Check the form")
    else:
        # NOTA: mode="json" convierte los enums en sus strings ("mobile_app"),
        # que es exactamente lo que espera la API en el body JSON.
        payload = request.model_dump(mode="json")
        params = {"prompt_version": prompt_version}
        started = time.perf_counter()
        try:
            with st.spinner("Estimating..."):
                response = httpx.post(
                    ESTIMATE_ENDPOINT,
                    json=payload,
                    params=params,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                )
                response.raise_for_status()
                elapsed_s = time.perf_counter() - started
                # NOTA: segunda llamada, sin coste de tokens: pide el prompt
                # renderizado para enseñarlo en el sidebar ("Prompt sent").
                preview = httpx.post(PREVIEW_ENDPOINT, json=payload, params=params, timeout=10.0)
                preview.raise_for_status()
        except httpx.HTTPError as exc:
            st.error(_error_detail(exc), title=f"Could not get an estimation from {API_BASE_URL}")
        else:
            result = response.json()
            st.session_state.result = result
            # Client-side wall time: what the user waited, cache hits included.
            st.session_state.last_metrics = {**result, "elapsed_s": elapsed_s}
            st.session_state.last_prompt = preview.json()

if st.session_state.result is not None:
    result = st.session_state.result
    with st.container(border=True):
        st.markdown(result["text"])
    st.caption(f"Prompt version `{result['prompt_version']}`")


# NOTA: el sidebar se dibuja AL FINAL del script a propósito: así, en la misma
# ejecución en la que llega la respuesta, ya puede mostrar sus métricas.
with st.sidebar:
    st.header("Service")
    st.code(ESTIMATE_ENDPOINT, language="text")
    primary = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{primary}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")
    if st.button("Clear result"):
        st.session_state.result = None
        st.session_state.last_metrics = None
        st.session_state.last_prompt = None
        st.rerun()

    st.header("Last call")
    last = st.session_state.last_metrics
    if last is None:
        st.caption("Submit the form to see the metrics of the call here.")
    else:
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
            help="Measured in the browser session, from request to response.",
            border=True,
        )
        if last.get("cache_hit"):
            st.badge("Served from Redis cache", icon=":material/bolt:", color="green")

    st.header("Prompt sent")
    prompt = st.session_state.last_prompt
    if prompt is None:
        st.caption("The rendered system and user messages of the last call appear here.")
    else:
        with st.expander("System message", icon=":material/description:"):
            st.code(prompt["system"], language="markdown", wrap_lines=True, height=400)
        with st.expander("User message", icon=":material/person:"):
            st.code(prompt["user"], language="markdown", wrap_lines=True)
