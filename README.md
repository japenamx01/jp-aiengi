

```markdown
# AI Engineering Project - Estimador CAG & Wrapper Conversacional

Repositorio para el desarrollo y despliegue de sistemas inteligentes basados en Arquitecturas CAG (Cache-Augmented Generation) y patrones de diseño para wrappers de modelos de lenguaje, desarrollado en el marco del programa de AI Engineering.

---

## 📂 Estructura del Repositorio

El proyecto se organiza de la siguiente manera:

```text
.
├── streamlit_app.py          # Interfaz web conversacional (Sesión 3 - Streamlit + Streaming + CAG)
├── .env                      # Fichero con las variables de entorno (API Key)
├── pyproject.toml / uv.lock  # Dependencias y gestión del entorno con uv
└── README.md                 # Documentación del proyecto

```

---

## 🚀 Funcionalidades Principales

1. **Estimador de Software Basado en CAG (Cache-Augmented Generation):**
* Inyección de contexto histórico de referencia para guiar al modelo en la generación de requerimientos y desgloses de fases de desarrollo.


2. **Interfaz Conversacional Web (Sesión 3):**
* **Nivel 1:** Chat interactivo mediante Streamlit (`st.chat_message`, `st.chat_input`) con persistencia del historial de la conversación (`st.session_state`).
* **Nivel 2:** Respuesta del LLM en tiempo real mediante streaming token a token (`st.write_stream`).
* **Nivel 3:** Panel de control lateral (`st.sidebar`) que muestra el System Prompt activo, el contexto estático inyectado y métricas de rendimiento (modelo, tokens de entrada/salida y tiempo de respuesta).


3. **Gestión Segura de Credenciales:**
* Carga centralizada de la clave de API mediante variables de entorno desde el fichero `.env` en la raíz.



---

## 🛠️ Ejecución Local con `uv`

### 1. Configuración del Fichero `.env`

Asegúrate de tener un fichero `.env` en la raíz de tu proyecto con tu clave de API:

```env
OPENAI_API_KEY=tu-clave-real-aqui

```

### 2. Ejecución de la Interfaz

Arranca la aplicación web directamente con `uv`:

```bash
uv run streamlit run streamlit_app.py

```

Esto abrirá de forma automática la interfaz gráfica en tu navegador web por defecto (`http://localhost:8501`).

```

```
