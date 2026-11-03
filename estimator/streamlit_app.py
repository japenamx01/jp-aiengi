"""Streamlit chat UI for the estimator.

Streamlit acts as an HTTP client of the FastAPI service: it POSTs to
``/api/v1/estimate/stream`` and renders the SSE chunks live with
``st.write_stream``. The endpoint URL is read from ``ESTIMATOR_API_BASE_URL``
(loaded from the same ``.env`` as the API), so the same UI works against a
local uvicorn or against docker-compose.

The sidebar shows the CAG context fetched from ``GET /api/v1/context`` and the
metrics of the last call, taken from the ``metrics`` SSE event.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
STREAM_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate/stream"
CONTEXT_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/context"

st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption(
    "Paste a meeting transcription. The answer streams token by token from the "
    "FastAPI service over Server-Sent Events."
)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_context() -> dict[str, Any] | None:
    """Fetch the system prompt and CAG examples; ``None`` if the API is down.

    Short TTL so a prompt change in the API (uvicorn ``--reload``) shows up
    without restarting Streamlit.
    """
    try:
        response = httpx.get(CONTEXT_ENDPOINT, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return response.json()


def stream_estimation(transcription: str, metrics: dict[str, Any]) -> Iterator[str]:
    """POST to the SSE endpoint and yield text chunks as they arrive.

    Per the SSE spec, a single message with internal newlines is serialised as
    multiple ``data:`` lines, and the client must join them with ``\\n`` to
    reconstruct the original payload. A blank line terminates the message.

    The ``metrics`` event sent before ``done`` is merged into ``metrics``.
    """
    payload = {"transcription": transcription}
    with httpx.stream(
        "POST",
        STREAM_ENDPOINT,
        json=payload,
        timeout=httpx.Timeout(120.0, connect=10.0),
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()
        current_event = "token"
        data_lines: list[str] = []
        for raw_line in response.iter_lines():
            if raw_line == "":
                if data_lines:
                    payload_text = "\n".join(data_lines)
                    data_lines = []
                    if current_event == "token":
                        yield payload_text
                    elif current_event == "error":
                        yield f"\n\n[error] {payload_text}"
                    elif current_event == "metrics":
                        metrics.update(json.loads(payload_text))
                    elif current_event == "done":
                        return
                current_event = "token"
                continue
            if raw_line.startswith("event:"):
                current_event = raw_line[6:].strip()
            elif raw_line.startswith("data:"):
                # The SSE spec defines exactly one space after `data:` as
                # framing, not payload — preserve any further whitespace.
                data_lines.append(
                    raw_line[6:] if raw_line.startswith("data: ") else raw_line[5:]
                )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Paste your meeting transcription here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        call_metrics: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            for chunk in stream_estimation(prompt, call_metrics):
                full_response += chunk
                placeholder.markdown(full_response + "▍")
            placeholder.markdown(full_response)
            if call_metrics:
                # Client-side wall time: what the user waited, cache hits included.
                call_metrics["elapsed_s"] = time.perf_counter() - started
                st.session_state.last_metrics = call_metrics
        except httpx.HTTPError as exc:
            full_response = f"Could not reach the estimator at `{STREAM_ENDPOINT}`: {exc}"
            placeholder.error(full_response)
        response = full_response

    st.session_state.messages.append({"role": "assistant", "content": response})


with st.sidebar:
    st.header("Service")
    st.code(STREAM_ENDPOINT, language="text")
    primary = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{primary}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.last_metrics = None
        st.rerun()

    st.header("Last call")
    last = st.session_state.last_metrics
    if last is None:
        st.caption("Send a transcription to see its metrics here.")
    else:
        usage = last.get("usage", {})
        # Plain text: model ids are too long for st.metric's large font.
        st.markdown(f"**Model:** `{last.get('model') or 'unknown'}`")
        tokens_in, tokens_out = st.columns(2)
        tokens_in.metric("Input tokens", f"{usage.get('input_tokens', 0):,}", border=True)
        tokens_out.metric("Output tokens", f"{usage.get('output_tokens', 0):,}", border=True)
        st.metric(
            "Response time",
            f"{last['elapsed_s']:.2f} s",
            help="Measured in the browser session, from request to last token.",
            border=True,
        )
        if last.get("cache_hit"):
            st.badge("Served from Redis cache", icon=":material/bolt:", color="green")

    st.header("CAG context")
    context = fetch_context()
    if context is None:
        st.warning(f"Could not load the context from `{CONTEXT_ENDPOINT}`.")
    else:
        with st.expander("System prompt", icon=":material/description:"):
            st.code(context["system_prompt"], language="markdown", wrap_lines=True, height=400)
        st.caption(f"{len(context['examples'])} reference estimations injected into the prompt:")
        for i, example in enumerate(context["examples"], start=1):
            with st.expander(f"{i}. {example['title']}", icon=":material/folder_open:"):
                st.markdown(f"**Meeting summary**\n\n{example['meeting_summary']}")
                st.markdown(example["estimation_markdown"])
