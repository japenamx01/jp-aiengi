import os
import textwrap
import time
import streamlit as st
from dotenv import load_dotenv

# Cargar las variables de entorno desde el fichero .env en la raíz
load_dotenv()

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Estimador CAG - Sesión 3",
    page_icon="🤖",
    layout="wide",
)

# ==========================================
# INICIALIZACIÓN Y LÓGICA CAG DEL PROYECTO
# ==========================================
# Lectura estricta de la variable de entorno desde .env.
# Se comprueba antes de importar el servicio: app.config falla al importarse sin la API key.
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ No se ha encontrado la variable `OPENAI_API_KEY` en el fichero `.env`.")
    st.stop()

# Mismo system prompt, ejemplos y llamada que el endpoint CAG de FastAPI
from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services.llm_service import build_system_prompt, stream_estimation

# Historial de conversación en Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Métricas de la última llamada (Nivel 3)
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {
        "model": settings.LLM_MODEL,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "response_time_sec": 0.0,
    }

# ==========================================
# PANEL LATERAL - CONTEXTO CAG Y MÉTRICAS (Nivel 3)
# ==========================================
with st.sidebar:
    st.header("🛠️ Panel de Control CAG")
    
    st.subheader("System Prompt Activo")
    st.code(build_system_prompt(), language="markdown", wrap_lines=True)
    
    st.subheader("Contexto Estático Inyectado")
    for ex in ESTIMATION_EXAMPLES:
        with st.expander(ex["meeting_summary"]):
            st.markdown(textwrap.dedent(ex["estimation"]))
    
    st.divider()
    
    st.subheader("📊 Métricas de la última llamada")
    metrics = st.session_state.last_metrics
    st.metric("Modelo", metrics["model"])
    st.metric("Tokens de Entrada", metrics["prompt_tokens"])
    st.metric("Tokens de Salida", metrics["completion_tokens"])
    st.metric("Tiempo de Respuesta", f"{metrics['response_time_sec']:.2f} s")

# ==========================================
# INTERFAZ PRINCIPAL DE CHAT (Niveles 1 y 2)
# ==========================================
st.title("💬 Estimador de Software con Arquitectura CAG")
st.markdown(
    "Pega la transcripción de tu reunión con el cliente para generar la estimación técnica en tiempo real."
)

# Mostrar el historial de mensajes al recargar la app
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de texto del usuario
if prompt := st.chat_input("Pega aquí la transcripción de la reunión..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta del asistente con Streaming (Nivel 2)
    with st.chat_message("assistant"):
        start_time = time.time()
        
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        try:
            stream = stream_estimation(history)

            def stream_response():
                full_content = ""
                completion_tokens_count = 0
                prompt_tokens_count = 0
                
                for chunk in stream:
                    if chunk.usage:
                        completion_tokens_count = chunk.usage.completion_tokens
                        prompt_tokens_count = chunk.usage.prompt_tokens
                        
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        full_content += content_chunk
                        yield content_chunk
                
                elapsed_time = time.time() - start_time
                st.session_state.last_metrics = {
                    "model": settings.LLM_MODEL,
                    "prompt_tokens": prompt_tokens_count,
                    "completion_tokens": completion_tokens_count,
                    "response_time_sec": elapsed_time,
                }

            response_text = st.write_stream(stream_response())
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            # Re-ejecutar para que el panel lateral muestre las métricas de esta llamada
            st.rerun()

        except Exception as e:
            st.error(f"Error al conectar con el servicio de LLM: {e}")
