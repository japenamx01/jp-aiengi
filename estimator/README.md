# Estimator CAG - Servicio de Estimacion de Software con IA

Servicio de estimacion de proyectos de software impulsado por IA, utilizando una arquitectura **Cache Augmented Generation (CAG)**.

## Que es CAG y por que lo usamos

CAG (Cache Augmented Generation) es un patron de arquitectura donde el contexto relevante se inyecta directamente en el prompt del LLM como texto estatico. En esta fase del proyecto, las estimaciones de referencia se incluyen como ejemplos dentro del prompt del sistema, sin necesidad de una base de datos vectorial ni busqueda semantica.

Este enfoque es ideal para empezar porque:
- Es simple de implementar y depurar
- No requiere infraestructura adicional (ni embeddings, ni vector stores)
- Funciona bien cuando el volumen de contexto es manejable (pocos ejemplos)

En modulos posteriores del master, este servicio evolucionara a una arquitectura **RAG** (Retrieval Augmented Generation) con base de datos vectorial para manejar un volumen mayor de ejemplos.

## Requisitos previos

- **Docker** y **Docker Compose** instalados
- Una **API key** de OpenAI o Anthropic
- Python **NO** es necesario localmente — todo se ejecuta dentro del contenedor

## Inicio rapido con Docker (recomendado)

1. Clonar el repositorio y entrar al directorio:
   ```bash
   cd estimator
   ```

2. Copiar el archivo de variables de entorno y configurar las API keys:
   ```bash
   cp .env.example .env
   # Editar .env y poner tu API key real
   ```

3. Construir y levantar el servicio:
   ```bash
   docker compose up --build
   ```

4. El servicio estara disponible en `http://localhost:8000`

## Alternativa: ejecucion local sin Docker

```bash
uv sync
# Configurar .env con tus API keys
uv run uvicorn app.main:app --reload
```

## Probar el servicio

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "The client wants to build a mobile app for managing restaurant reservations. They need user registration, a restaurant search with filters by cuisine and location, a real-time reservation system with availability checking, push notifications for reservation confirmations and reminders, and an admin panel for restaurant owners to manage their listings and view analytics."
  }'
```

## Estructura del proyecto

```
estimator/
├── app/
│   ├── main.py                  # Aplicacion FastAPI, health check, CORS
│   ├── config.py                # Configuracion con Pydantic Settings
│   ├── dependencies.py          # Singletons de cache Redis y LLMWrapper (Depends)
│   ├── routers/
│   │   └── estimations.py       # POST /estimate, POST /estimate/stream (SSE + evento
│   │                            #   metrics) y GET /context (system prompt y ejemplos)
│   ├── services/
│   │   ├── llm_service.py       # Construccion del system prompt y orquestacion
│   │   ├── llm_wrapper.py       # LiteLLM Router: fallback primario -> secundario,
│   │   │                        #   cache, coste y uso de tokens en streaming
│   │   ├── cache.py             # Cache exact-match en Redis
│   │   └── evaluation.py        # Validacion estructural de la estimacion
│   ├── schemas/
│   │   └── estimation.py        # Modelos Pydantic (request/response, contexto CAG)
│   ├── context/
│   │   └── examples.py          # Ejemplos de estimacion (contexto CAG)
│   ├── fixtures/                # Transcripciones de ejemplo (corta y larga)
│   └── static/
│       └── sse_demo.html        # Demo HTML del endpoint SSE
├── tests/
│   ├── conftest.py              # Fixture compartida (TestClient de FastAPI)
│   ├── test_health.py           # Health check
│   ├── test_estimate_endpoint.py  # POST /estimate
│   ├── test_estimate_stream.py  # Streaming SSE, evento metrics y GET /context
│   ├── test_llm_wrapper.py      # Fallback del Router, cache, coste y metricas
│   ├── test_cache.py            # Cache Redis (fakeredis)
│   ├── test_evaluation.py       # Validacion estructural
│   └── test_examples_format.py  # Formatos de los ejemplos (markdown/json/narrativo)
├── streamlit_app.py             # UI de chat + sidebar (system prompt, ejemplos CAG
│                                #   y metricas de la ultima llamada)
├── Dockerfile                   # Build multi-stage con uv (API y Streamlit)
├── docker-compose.yml           # API, Streamlit y Redis para desarrollo local
├── .env.example                 # Variables de entorno de referencia
└── pyproject.toml               # Dependencias y configuracion
```

## Documentacion interactiva

Con el servicio corriendo, accede a la documentacion Swagger UI en:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Sesion 3 — LiteLLM, Redis cache, SSE y Streamlit

A partir de la Sesion 3 el servicio incorpora una capa de wrapper sobre el LLM que anade:

- **Fallback de proveedor** (LiteLLM Router) — si el modelo primario falla, se intenta el secundario
- **Cache exact-match** en Redis — la misma transcripcion no vuelve a pagar tokens
- **Streaming SSE** — endpoint `POST /api/v1/estimate/stream` que emite los tokens segun llegan
- **UI Streamlit** — cliente real que consume el endpoint SSE

### Fallback real: un grupo por modelo en el Router

El Router de LiteLLM agrupa los despliegues por `model_name`. Si el primario y el
fallback comparten nombre, el Router los trata como **intercambiables** y la estrategia
por defecto (`simple-shuffle`) reparte el trafico entre ambos: en una prueba de 200
llamadas, 87 fueron a Claude aunque OpenAI respondia bien. Eso no es un fallback, es
un balanceo de carga (y Claude Haiku cuesta ~7x mas por token que `gpt-4o-mini`).

Para que el secundario se use **solo cuando el primario falla**, cada modelo vive en su
propio grupo y `fallbacks` enlaza uno con otro (`app/services/llm_wrapper.py`):

```python
PRIMARY_GROUP = "estimator"
FALLBACK_GROUP = "estimator-fallback"

Router(
    model_list=[
        {"model_name": PRIMARY_GROUP, "litellm_params": {"model": primary_model, ...}},
        {"model_name": FALLBACK_GROUP, "litellm_params": {"model": fallback_model, ...}},
    ],
    fallbacks=[{PRIMARY_GROUP: [FALLBACK_GROUP]}],
    num_retries=num_retries,
)

# Todas las llamadas van al grupo primario; LiteLLM salta al fallback si falla
self.router.completion(model=PRIMARY_GROUP, ...)
```

Resultado: el 100% de las llamadas sanas las responde `PRIMARY_MODEL`, y con OpenAI
caido (p. ej. una API key invalida) responde `FALLBACK_MODEL`. El modelo que
realmente respondio aparece en el sidebar de Streamlit y en el log `llm_call_completed`
/ `llm_stream_completed`. El test
`test_router_always_picks_primary_and_falls_back_on_failure` evita que se vuelvan a
juntar en un mismo grupo.

Para que el fallback funcione hace falta `ANTHROPIC_API_KEY` en `.env`; sin ella la
app arranca igual pero no tiene respaldo.

### Arrancar la stack completa

```bash
cd estimator
docker compose up --build
# La API queda en http://localhost:8000, Streamlit en http://localhost:8501
# y Redis en redis://localhost:6379
```

### Probar el endpoint SSE

Demo HTML: abrir [http://localhost:8000/static/sse_demo.html](http://localhost:8000/static/sse_demo.html).

Desde CLI:
```bash
curl -N -X POST http://localhost:8000/api/v1/estimate/stream \
  -H 'Content-Type: application/json' \
  -d '{"transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks."}'
```

### Verificar la cache

```bash
# La misma peticion dos veces — la segunda devuelve cache_hit: true
curl -s localhost:8000/api/v1/estimate -H 'Content-Type: application/json' \
  -d '{"transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks."}' \
  | jq '{cache_hit, cost_usd}'

# Inspeccionar las claves en Redis
docker compose exec redis redis-cli KEYS 'estimation:*'
```

### Streamlit

Streamlit levanta con `docker compose up` como el servicio `streamlit` y consume el
endpoint SSE por HTTP. Abrir [http://localhost:8501](http://localhost:8501).
Dentro de la red de compose apunta a `http://estimator:8000`.

Para correrlo **fuera** de Docker (contra la API en `localhost:8000`):

```bash
cd estimator
uv sync
uv run streamlit run streamlit_app.py
```

La URL del backend se lee de `ESTIMATOR_API_BASE_URL` (default `http://localhost:8000`).

---

> Este proyecto forma parte del **Master en AI Engineering** y servira como base para evolucionar hacia una arquitectura RAG con base de datos vectorial en modulos posteriores.
