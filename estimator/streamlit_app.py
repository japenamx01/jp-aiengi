"""Streamlit entry point: two pages over the same estimator API.

- ``ui/quick_estimate.py`` — the typed form of Session 4: one request, no memory.
- ``ui/conversation.py`` — Session 5: a session with sliding-window history,
  project metadata and PDF/Word attachments.

Run with ``uv run streamlit run streamlit_app.py``. The API location is read from
``ESTIMATOR_API_BASE_URL`` (see ``ui/common.py``).

NOTA: con ``st.navigation`` este archivo solo decide QUÉ página se ejecuta.
Streamlit pinta el menú en el sidebar y, en cada re-ejecución, corre este
script y después el de la página elegida. ``st.set_page_config`` va aquí, una
sola vez, para que valga para todas las páginas. ``st.session_state`` es
compartido entre páginas: por eso la conversación usa claves ``conv_*``.
"""

import streamlit as st

st.set_page_config(page_title="Software Estimator", page_icon="📊")

page = st.navigation(
    [
        st.Page("ui/conversation.py", title="Conversation", icon=":material/forum:", default=True),
        st.Page("ui/quick_estimate.py", title="Quick estimate", icon=":material/calculate:"),
    ]
)
page.run()
