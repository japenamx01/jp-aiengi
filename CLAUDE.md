# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This repo currently contains a single project under `estimator/` — all commands below assume `cd estimator` first. The repo is part of a Master en AI Engineering and is intended to evolve module-by-module (CAG → RAG with vector DB in later modules).

## Common commands

Dependency / runtime management uses **uv** (Astral) and Python 3.11.

```bash
# Install deps (creates .venv)
uv sync

# Run the API locally with hot reload
uv run uvicorn app.main:app --reload

# Run the full test suite
uv run pytest

# Run a single test file or test
uv run pytest tests/test_health.py
uv run pytest tests/test_health.py::test_name -v

# Lint
uv run ruff check .
uv run ruff format .

# Docker (recommended dev path — bind-mounts app/ for live reload)
docker compose up --build
```

Service listens on `http://localhost:8000`; `/docs` (Swagger) and `/redoc` are enabled. Health probe at `GET /health`. Main API endpoint is `POST /api/v1/estimate`.

## Architecture

The estimator is a FastAPI service implementing **Cache Augmented Generation (CAG)**: reference estimations are inlined as static text inside the system prompt — no vector store, no retrieval step. This is a deliberate first-stage choice; later modules will migrate to RAG.

Request flow (Session 4):

1. `streamlit_app.py` — a `st.form` builds an `EstimationRequest` (it imports the same Pydantic model from `app.schemas.estimation`) and POSTs it to `/api/v1/estimate?prompt_version=...`.
2. `app/routers/estimations.py` — validates the typed body (`description` 20-2000 chars + enums `project_type`, `detail_level`, `output_format`), calls `render_estimation_prompt`, and sends `system` and `user` as **separate** messages through the `LLMWrapper` (`Depends(get_llm_wrapper)`). Returns `EstimationResponse` (`text`, `prompt_version` + call metadata for the sidebar). Also: `POST /estimate/stream` (same contract over SSE: `token`, `metrics`, `done`), `POST /prompt/preview` (rendered messages, no LLM call), `GET /prompts/versions`.
3. `app/prompts/loader.py::render_estimation_prompt(request, version="v1") -> (system, user)` — Jinja2 `Environment` with `StrictUndefined`, `trim_blocks`, `lstrip_blocks`. Templates live in `app/prompts/estimation/<version>/{system,user,examples}.j2`. Versions are whitelisted against the directories on disk; unknown ones raise `UnknownPromptVersionError` (the router maps it to 422). Each render logs `prompt_rendered` with the version and a content hash.
4. `app/services/llm_wrapper.py` — LiteLLM Router with cache, fallback, cost and token usage (also in streaming via `stream_options.include_usage`).

Key design points future changes should respect:

- **Prompts are versioned and immutable.** Never edit a published `vN/` in place: copy it to `vN+1/` and change it there, so every logged `prompt_version` maps to the exact text used. `system.j2` includes its own version's `examples.j2` by full path.
- **Prompt text lives only in `.j2` files** — no f-string prompts in Python. Pricing assumptions (62.50 EUR/h dev, 50 EUR/h designer) are in `system.j2`.
- **Few-shot examples are data rendered per format**: `examples.j2` defines the examples once (`{% set examples = [...] %}`) and renders them in the requested `output_format`/`detail_level`, so they never contradict the format instructions. The template tests rely on this (e.g. `confidence_pct` must not appear with `narrative`). Example costs must equal the sum of their phases.
- **Router fallback needs one model group per model** (`PRIMARY_GROUP` / `FALLBACK_GROUP` linked by `fallbacks`). Putting both deployments under the same `model_name` makes LiteLLM load-balance between them. `test_router_always_picks_primary_and_falls_back_on_failure` guards this.
- **Cache key = rendered prompt + generation knobs**, so changing a form parameter or the prompt version invalidates the cache automatically.
- **Settings are a cached singleton** via `app/config.py::get_settings` (`@lru_cache`); any change to `.env` requires restarting the service (a `--reload` is not enough). At least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` must be set; the fallback needs both.
- **Logging** is `structlog`, configured in `lifespan` (`main.py`): JSON in `production`, console in dev. Use `structlog.get_logger()` rather than stdlib `logging`.

## Tests

- `tests/prompts/` — template tests: they assert on the rendered text only (no LLM, no network, milliseconds). They do not evaluate the quality of the model's answer.
- Endpoint and stream tests replace the wrapper with a stub via `app.dependency_overrides[get_llm_wrapper]`; wrapper and cache tests use `fakeredis`.

## Configuration

`.env` (copied from `.env.example`) drives everything via `pydantic-settings`. Notable vars:

- `PRIMARY_MODEL` / `FALLBACK_MODEL` — LiteLLM model ids for the Router (defaults `gpt-4o-mini` / `claude-haiku-4-5-20251001`).
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — at least one is required; the fallback needs both.
- `REDIS_URL`, `CACHE_TTL` — exact-match cache (compose overrides `REDIS_URL` to `redis://redis:6379`).
- `ESTIMATOR_API_BASE_URL` — where Streamlit reaches the API (compose sets `http://estimator:8000`).
- `APP_ENV` — `development` | `staging` | `production` (controls log renderer).

## Docker

Multi-stage Dockerfile: `builder` installs prod-only deps with `uv sync --no-install-project --no-dev`, `runtime` is a clean `python:3.11-slim` that only carries `/app/.venv`, `app/` and `streamlit_app.py`, runs as non-root `appuser`. There is a Docker-native HEALTHCHECK against `/health`. `docker-compose.yml` runs three services from that image plus Redis: `estimator` (API, :8000), `streamlit` (UI, :8501) and `redis`. It bind-mounts `./app` (and `streamlit_app.py`) and adds `--reload` for dev — strip both for any production deployment.
