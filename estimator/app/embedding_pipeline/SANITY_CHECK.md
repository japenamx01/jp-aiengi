# Sanity check de los embeddings

- **Modelo:** `text-embedding-3-small`, 1536 dimensiones (por defecto, sin recortar).
- **Herramienta:** `scripts/compare.py`. Reutiliza `OpenAIEmbedder` y calcula la coseno a mano con `math`.
- **Fecha:** 2026-11-24.
- **Reproducible:** el mismo valor sale dentro del contenedor (`docker compose exec estimator python scripts/compare.py ...`) y fuera (`uv run python scripts/compare.py ...`).

## Resultados

| Pareja | Texto 1 | Texto 2 | Coseno | Expectativa | ¿Cumple? |
|---|---|---|---|---|---|
| A — cercanos | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Authorization service using JSON Web Tokens for a banking application" | **0.5957** | alta (> 0.6) | casi, se queda 0.004 por debajo |
| B — no relacionados | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Database migration from MySQL to PostgreSQL with zero downtime" | **0.1920** | baja (< 0.4) | sí |
| C — genéricos | "Backend services" | "API development" | **0.5407** | sin expectativa fija | — |

## Comentario

El modelo **discrimina bien**: entre A (0.60) y B (0.19) hay 0.40 de distancia, así que el pipeline funciona de punta a punta y separa «auth para banca» de «migración de BD».

Lo que llama la atención:

- **A se queda justo por debajo de 0.6** aunque describe lo mismo. Los dos textos casi no comparten palabras ("OAuth/JWT/fintech" frente a "Authorization/JSON Web Tokens/banking") y el modelo tiene que unirlos solo por significado. Los umbrales dependen del modelo y de la redacción: no son constantes universales.
- **C (0.54) queda casi tan alta como A**, con dos frases genéricas de dos palabras. Un texto corto y vago tiene poca señal propia y cae en una zona difusa tipo «cosas de backend». Lo que importa es la **posición relativa** (ranking), no el valor absoluto: un 0.54 no significa «54 % de parecido».

## Experimento extra: ¿sirve la cabecera de contexto?

C es la razón de que el chunker anteponga la cabecera del presupuesto padre (proyecto, sector, año, tecnología) a cada componente. Para comprobarlo con números, `scripts/header_effect.py` compara componentes casi iguales de sectores distintos. Para cada consulta mide la coseno contra el componente correcto (*target*) y contra el señuelo de otro sector (*decoy*), con cabecera y sin ella. El *gap* es target − decoy: cuanto más alto, mejor separa.

| Consulta | Target / decoy | Sin cabecera (gap) | Con cabecera (gap) |
|---|---|---|---|
| "Inventory management for a hospital pharmacy" | STOCK-002 (healthcare) / INV-003 (ecommerce) | 0.4614 / 0.4204 (**+0.041**) | 0.5964 / 0.3803 (**+0.216**) |
| "Sensor data ingestion for a factory production line" | TELE-001 (industrial) / INGEST-001 (healthcare) | 0.5288 / 0.5430 (**−0.014**, gana el señuelo) | 0.5769 / 0.4634 (**+0.114**) |
| "Onboarding and risk checks for new clients of an investment firm" | ONB-003 (finance) / VENDOR-001 (ecommerce) | 0.5936 / 0.4636 (**+0.130**) | 0.5647 / 0.3915 (**+0.173**) |

Qué se lee en la tabla:

- **Sin cabecera, el caso 2 sale al revés:** la ingesta de *wearables* de un hospital queda más cerca de «sensores de fábrica» que la telemetría industrial correcta. Con la cabecera (`Client sector: industrial`) el orden se corrige.
- **En los tres casos el gap crece con la cabecera.** Es justo el problema que resuelven los *contextual chunk headers*.
- **El precio:** en el caso 3 la coseno del target *baja* (0.59 → 0.56), porque la cabecera añade texto que la consulta no menciona y diluye un poco el vector. Aun así separa mejor. Es material para el directo: cuánto contexto añadir y cuándo deja de ayudar.

Esto no es una validación formal (no hay recall@k ni un dataset de consultas). Es el mínimo que muestra que los vectores tienen sentido y que la decisión de diseño del chunker se nota.
