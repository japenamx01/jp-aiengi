import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

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
# CONSTANTES Y CONFIGURACIÓN CAG (Nivel 3)
# ==========================================
SYSTEM_PROMPT = (
    "Eres un asistente experto en ingeniería de software y estimación de proyectos. "
    "Tu objetivo es analizar transcripciones de reuniones de clientes, extraer los requerimientos "
    "técnicos y funcionales, y generar una estimación estructurada de esfuerzo, tiempos y desglose "
    "de fases de desarrollo basándote en el contexto CAG provisto."
)

STATIC_CONTEXT = """
[Ejemplo Histórico de Referencia - Proyecto de Migración de Base de Datos]
- Alcance: Migración de MS SQL Server 2012 a EnterpriseDB (EPAS 18).
- Esfuerzo estimado: 120 horas distribuidas en 3 semanas.
- Fases: 
  1. Auditoría de esquemas y dependencias (20h)
  2. Adaptación de stored procedures y sintaxis (50h)
  3. Pruebas de integración y validación con pgvector (30h)
  4. Despliegue en producción y tunning (20h)
"""

# ==========================================
# INICIALIZACIÓN DEL CLIENTE Y ESTADO
# ==========================================
# Lectura estricta de la variable de entorno desde .env
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ No se ha encontrado la variable `OPENAI_API_KEY` en el fichero `.env`.")
    st.stop()

client = OpenAI(api_key=api_key)

# Historial de conversación en Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Métricas de la última llamada (Nivel 3)
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {
        "model": "gpt-4o-mini",
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
    st.code(SYSTEM_PROMPT, language="markdown")
    
    st.subheader("Contexto Estático Inyectado")
    st.text_area(
        "Base de Conocimiento CAG",
        value=STATIC_CONTEXT.strip(),
        height=180,
        disabled=True,
    )
    
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
        
        messages_payload = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContexto de referencia:\n{STATIC_CONTEXT}"}
        ]
        for m in st.session_state.messages:
            messages_payload.append({"role": m["role"], "content": m["content"]})

        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_payload,
                stream=True,
                stream_options={"include_usage": True}
            )

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
                    "model": "gpt-4o-mini",
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
