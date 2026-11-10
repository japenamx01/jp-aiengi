"""Conversation page (Session 5): multi-turn estimation with memory and attachments.

On load the page creates a session (``POST /api/v1/sessions``) and keeps its
``session_id`` in ``st.session_state``. Each turn sends the transcript, the
typed parameters and the attached PDF/Word files as ``multipart/form-data`` to
``POST /api/v1/sessions/{session_id}/estimate``. The sidebar shows the project
metadata the service has accumulated, separate from the message history.

NOTA: dos memorias distintas en esta página
-------------------------------------------
  - ``st.session_state.conv_turns`` es solo para PINTAR la conversación en
    pantalla (Streamlit re-ejecuta el script y sin esto se borraría).
  - La memoria que usa el modelo vive en el SERVIDOR (historial con ventana +
    project_metadata). El cliente solo guarda el ``session_id`` para que el
    servidor sepa de qué conversación se trata.
Por eso la pantalla puede mostrar 8 turnos aunque el modelo solo reciba los
últimos 6: lo ves en el indicador "History window" del sidebar.
"""

from __future__ import annotations

import time

import httpx
import streamlit as st

from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType
from app.schemas.session import TRANSCRIPT_MAX_CHARS
from ui.common import (
    API_BASE_URL,
    LABELS,
    SESSIONS_ENDPOINT,
    error_detail,
    render_call_metrics,
    render_service_info,
)


def start_conversation() -> None:
    """Create a new server-side session and reset the page state."""
    # NOTA: "New conversation" NO borra nada en el servidor: crea otra sesión
    # vacía. La anterior queda huérfana en el diccionario hasta que el servicio
    # se reinicie (aceptable en esta fase; con persistencia habría expiración).
    response = httpx.post(SESSIONS_ENDPOINT, timeout=10.0)
    response.raise_for_status()
    created = response.json()
    st.session_state.conv_session_id = created["session_id"]
    st.session_state.conv_max_turns = created["max_turns"]
    st.session_state.conv_turns = []
    st.session_state.conv_metadata = None
    st.session_state.conv_window = 0
    st.session_state.conv_last = None
    st.session_state.conv_lost = False


def send_turn(transcript: str, uploads: list, params: dict) -> dict:
    """POST one turn as multipart/form-data and return the JSON answer."""
    # NOTA: cada archivo viaja como una tupla (nombre, bytes, content-type)
    # bajo el MISMO nombre de campo "attachments": así FastAPI los recibe como
    # list[UploadFile]. Los parámetros tipados van en ``data`` (campos Form).
    files = [
        ("attachments", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in uploads
    ]
    response = httpx.post(
        f"{SESSIONS_ENDPOINT}/{st.session_state.conv_session_id}/estimate",
        data={"transcript": transcript, **params},
        files=files or None,
        timeout=httpx.Timeout(180.0, connect=10.0),
    )
    response.raise_for_status()
    return response.json()


# NOTA: se crea la sesión una sola vez por pestaña del navegador: en las
# re-ejecuciones siguientes la clave ya existe y no se vuelve a llamar.
if "conv_session_id" not in st.session_state:
    try:
        start_conversation()
    except httpx.HTTPError as exc:
        st.error(error_detail(exc), title=f"Could not create a session at {API_BASE_URL}")
        st.stop()

st.title("Conversation")
st.caption(
    "Refine the same project turn by turn and attach PDF or Word documents. The service "
    "keeps the last turns and a separate memory of the project facts."
)

# Reserved above the form so the turns appear in reading order.
chat = st.container()

# NOTA: clear_on_submit=True vacía el texto y el selector de archivos tras
# enviar: cada adjunto se manda UNA vez; después su contenido ya forma parte
# del historial (y sus datos clave, de la memoria).
with st.form("turn_form", clear_on_submit=True):
    transcript = st.text_area(
        "Message",
        height=120,
        max_chars=TRANSCRIPT_MAX_CHARS,
        placeholder="E.g. A project called VetCare: online bookings for 12 vet clinics.",
    )
    uploads = st.file_uploader(
        "Attachments",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Specifications, previous proposals or architecture notes (PDF or .docx).",
    )
    with st.expander("Estimation settings", icon=":material/tune:"):
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
    submitted = st.form_submit_button("Send", type="primary", icon=":material/send:")

if submitted:
    if not transcript.strip():
        st.warning("Write a message before sending.", title="Empty message")
    else:
        params = {
            "project_type": project_type.value,
            "detail_level": detail_level.value,
            "output_format": output_format.value,
        }
        started = time.perf_counter()
        try:
            with st.spinner("Estimating and updating the project memory..."):
                answer = send_turn(transcript.strip(), uploads or [], params)
        except httpx.HTTPStatusError as exc:
            # NOTA: 404 = la sesión ya no existe en el servidor (se reinició y
            # el diccionario en memoria se vació). Es la volatilidad aceptada.
            if exc.response.status_code == 404:
                st.session_state.conv_lost = True
            else:
                st.error(error_detail(exc), title="The turn could not be estimated")
        except httpx.HTTPError as exc:
            st.error(error_detail(exc), title=f"Could not reach {API_BASE_URL}")
        else:
            st.session_state.conv_turns.append(
                {
                    "user": transcript.strip(),
                    "attachments": answer["attachments"],
                    "assistant": answer["text"],
                    "turn": answer["turn"],
                }
            )
            st.session_state.conv_metadata = answer["project_metadata"]
            st.session_state.conv_window = answer["turns_in_window"]
            st.session_state.conv_last = {**answer, "elapsed_s": time.perf_counter() - started}

if st.session_state.conv_lost:
    st.warning(
        "The service restarted and this conversation is gone (sessions live in memory). "
        "Start a new conversation from the sidebar.",
        title="Session not found",
    )

with chat:
    if not st.session_state.conv_turns:
        st.info(
            "Describe the project to get a first estimation, then refine it.",
            icon=":material/forum:",
        )
    for turn in st.session_state.conv_turns:
        with st.chat_message("user"):
            st.markdown(turn["user"])
            for attachment in turn["attachments"]:
                cut = " · truncated" if attachment["truncated"] else ""
                st.caption(
                    f":material/attach_file: {attachment['filename']} ({attachment['chars']:,} chars{cut})"
                )
        with st.chat_message("assistant"):
            st.markdown(turn["assistant"])
            st.caption(f"Turn {turn['turn']}")


with st.sidebar:
    st.header("Conversation")
    st.code(st.session_state.conv_session_id, language="text")
    if st.button("New conversation", icon=":material/add_comment:"):
        try:
            start_conversation()
        except httpx.HTTPError as exc:
            st.error(error_detail(exc), title="Could not create a session")
        else:
            st.rerun()

    # NOTA: este panel es la "memoria" (project_metadata). Compárala con la
    # conversación: aunque la ventana descarte turnos viejos, estos hechos
    # siguen aquí y se inyectan en el system prompt de cada turno.
    st.header("Project memory")
    metadata = st.session_state.conv_metadata
    if not metadata or not any(metadata.values()):
        st.caption("Nothing captured yet. Facts appear here after the first turn.")
    else:
        st.markdown(f"**Project name:** {metadata['project_name'] or '—'}")
        team = metadata["assumed_team_size"]
        st.markdown(f"**Assumed team size:** {f'{team} people' if team else '—'}")
        st.markdown("**Mentioned technologies:**")
        if metadata["mentioned_technologies"]:
            with st.container(horizontal=True):
                for tech in metadata["mentioned_technologies"]:
                    st.badge(tech, color="blue")
        else:
            st.caption("—")
        st.markdown(f"**Agreed scope:** {metadata['agreed_scope'] or '—'}")
        with st.expander("Raw JSON", icon=":material/data_object:"):
            st.json(metadata)

    # NOTA: el "historial" (lo que viaja en messages). Llega como mucho a
    # max_turns; a partir de ahí cada turno nuevo expulsa al más antiguo.
    st.header("History window")
    window, max_turns = st.session_state.conv_window, st.session_state.conv_max_turns
    st.progress(window / max_turns, text=f"{window} / {max_turns} turns kept")
    st.caption(
        f"The model receives the last {max_turns - 1} turns plus the new message. "
        "Older turns are dropped; the project memory keeps their facts."
    )

    st.header("Last call")
    if st.session_state.conv_last is None:
        st.caption("Send a message to see the metrics of the call here.")
    else:
        render_call_metrics(st.session_state.conv_last)

    render_service_info(SESSIONS_ENDPOINT)
