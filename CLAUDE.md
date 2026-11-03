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

Request flow:

1. `app/routers/estimations.py` — `POST /api/v1/estimate` accepts an `EstimationRequest` (transcription + optional knobs: `preprocessing`, `example_format`, `num_examples`, `use_examples`, `model`, `max_tokens`, `thinking_budget`, `evaluate`).
2. `app/services/llm_service.py::generate_estimation` — builds a `GenerationOptions`, optionally runs `extract_requirements` for two-phase preprocessing, builds the system prompt (`build_system_prompt`), and dispatches to either `_call_openai` or `_call_anthropic` based on `LLM_PROVIDER`. Returns a dict with `estimation`, `model`, `provider`, `usage`, `finish_reason`, `latency_ms`, `preprocessing`, `extracted_requirements`.
3. `app/services/evaluation.py::evaluate_estimation_structure` — pure regex/parsing pass that produces a `StructureCheck` (sections present, breakdown sums match declared totals, `finish_reason` ok, `score`, `issues`).
4. The router merges the LLM result with the validation and returns `EstimationResponse`.

Key design points future changes should respect:

- **Provider abstraction lives in one file** (`llm_service.py`). Both branches return the same dict shape (`estimation`, `model`, `provider`, `finish_reason`, `usage`) so the router stays provider-agnostic. Keep that contract when adding providers.
- **Settings are a cached singleton** via `app/config.py::get_settings` (`@lru_cache`). Pydantic Settings runs a `model_validator` that requires the API key matching `LLM_PROVIDER` to be set — changing provider also requires changing the key. Because of the cache, **any change to `.env` requires restarting uvicorn** (a `--reload` is not enough; the singleton is module-scoped).
- **CAG examples** live in `app/context/examples.py`. The single source of truth is `CANONICAL_EXAMPLES: list[CanonicalExample]`; the three formatters (`_format_markdown`, `_format_json`, `_format_narrative`) derive their output from it. `estimation_markdown` is precomputed per example so the Markdown rendering stays byte-for-byte stable. When this graduates to RAG, this module is the seam to replace.
- **Pricing assumptions** (62.50 EUR/h dev, 50 EUR/h designer) live in `build_system_prompt`. Edit there, not in examples. Each canonical example must satisfy `sum_h == total_hours` and `sum_c == total_cost` (a unit test enforces this).
- **Output prompt switch**: `llm_service.py` defines `PROMPT_OUTPUT_BASIC` and `PROMPT_OUTPUT_STRUCTURED`; the `ACTIVE_OUTPUT_PROMPT` line picks one. The Session 2 live demo flips this constant on the fly.
- **`thinking_budget` is Anthropic-only** — for OpenAI it is logged as a warning and ignored. The Anthropic wrapper auto-pads `max_tokens` so it stays above the budget.
- **Logging** is `structlog`, configured in `lifespan` (`main.py`): JSON in `production`, console in dev. Use `structlog.get_logger()` rather than stdlib `logging`.

## How to compare in live demos

The endpoint is built to support side-by-side comparisons without code changes. The `usage`, `latency_ms`, `finish_reason` and `validation.score` fields in the response are the projected metrics during sessions.

```bash
TRANS=$(jq -Rs . < estimator/app/fixtures/long_transcription.txt)

# Three preprocessing modes
for MODE in none inline_cleaning two_phase; do
  curl -s localhost:8000/api/v1/estimate -H 'Content-Type: application/json' \
    -d "{\"transcription\": $TRANS, \"preprocessing\":\"$MODE\"}" \
    | jq "{mode:\"$MODE\", score:.validation.score, latency_ms, usage}"
done

# Number of CAG examples (rendimiento decreciente)
for N in 0 1 3 5; do
  curl -s localhost:8000/api/v1/estimate -H 'Content-Type: application/json' \
    -d "{\"transcription\": $TRANS, \"num_examples\": $N}" \
    | jq "{n: $N, input_tokens: .usage.input_tokens, latency_ms, score: .validation.score}"
done
```

The full session script lives in `estimator/docs/session-2-guide.md`.

## Configuration

`.env` (copied from `.env.example`) drives everything via `pydantic-settings`. Notable vars:

- `LLM_PROVIDER` — `openai` (default) or `anthropic`.
- `LLM_MODEL` — model id passed straight through to the SDK (e.g. `gpt-4o-mini`, `claude-...`).
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — only the one matching `LLM_PROVIDER` is required.
- `APP_ENV` — `development` | `staging` | `production` (controls log renderer).

## Docker

Multi-stage Dockerfile: `builder` installs prod-only deps with `uv sync --no-install-project --no-dev`, `runtime` is a clean `python:3.11-slim` that only carries `/app/.venv` and `app/`, runs as non-root `appuser`. There is a Docker-native HEALTHCHECK against `/health`. `docker-compose.yml` bind-mounts `./app` and adds `--reload` for dev — strip both for any production deployment.
