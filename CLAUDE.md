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

Request flow (Session 4, stateless):

1. `ui/quick_estimate.py` (a page of `streamlit_app.py`, which uses `st.navigation`) — a `st.form` builds an `EstimationRequest` (it imports the same Pydantic model from `app.schemas.estimation`) and POSTs it to `/api/v1/estimate?prompt_version=...`.
2. `app/routers/estimations.py` — validates the typed body (`description` 20-2000 chars + enums `project_type`, `detail_level`, `output_format`), calls `render_estimation_prompt`, and sends `system` and `user` as **separate** messages through the `LLMWrapper` (`Depends(get_llm_wrapper)`). Returns `EstimationResponse` (`text`, `prompt_version` + call metadata for the sidebar). Also: `POST /estimate/stream` (same contract over SSE: `token`, `metrics`, `done`), `POST /prompt/preview` (rendered messages, no LLM call), `GET /prompts/versions`.
3. `app/prompts/loader.py::render_estimation_prompt(request, version="v1") -> (system, user)` — Jinja2 `Environment` with `StrictUndefined`, `trim_blocks`, `lstrip_blocks`. Templates live in `app/prompts/estimation/<version>/{system,user,examples}.j2`. Versions are whitelisted against the directories on disk; unknown ones raise `UnknownPromptVersionError` (the router maps it to 422). Each render logs `prompt_rendered` with the version and a content hash.
4. `app/services/llm_wrapper.py` — LiteLLM Router with cache, fallback, cost and token usage (also in streaming via `stream_options.include_usage`). `complete_messages(messages)` sends a whole conversation; `complete()` delegates to it with `[system, user]` and keeps the Session 4 cache key.

Conversational flow (Session 5), `app/routers/sessions.py`:

1. `POST /api/v1/sessions` creates a `Session` in `SessionStore` (in-process dict, `Depends(get_session_store)`) and returns a UUID4 `session_id`.
2. `POST /api/v1/sessions/{id}/estimate` takes `multipart/form-data`: `transcript`, optional `attachments: list[UploadFile]` and the typed enums as `Form` fields. Attachments are converted to text locally by `app/services/attachments.py` (pypdf / python-docx; unsupported type → 415, unreadable/too big → 422).
3. `render_session_prompt` renders `estimation/v3` with the current `ProjectMetadata` in `<project_metadata>` and the attachments between `--- attachment: name ---` markers. `ConversationHistory.to_messages_list(system, pending_user=user)` builds `[system] + last MAX_TURNS-1 turns + [new user]`.
4. After a successful call the turn is stored (a `deque(maxlen=MAX_TURNS)`) and `app/services/metadata_extractor.py` runs a second LLM call (`prompts/extraction/v1`) that returns JSON, validated and merged into the metadata (nulls never erase known facts; technologies accumulate). Extractor failures are logged and keep the old metadata. A provider failure (502) leaves history and metadata untouched.
5. `ui/conversation.py` creates the session on load, sends each turn with its files and shows the metadata, the window fill and the last-call metrics in the sidebar.

Stress test (Session 6), `evals/stress/`:

1. Every `POST /sessions/{id}/estimate` builds a `TurnObservation` (`app/schemas/session.py`) right before returning: the 13 fields of the exercise (`anchors_count`/`summary_chars` are always 0 and `last_resolved_tier` None — this CAG has no anchors, summarizer or tiers) plus extractor usage and `turn_cost_usd`. It is logged as one `turn_observed` event, returned as `observation` and exposed as `last_observation` by `GET /sessions/{id}`. On a cache hit the observation reports cost 0 and the measured latency.
2. `evals/stress/run.py` walks `scenarios.py` (3 × 20 turns with fact-trackers) × PDF sizes (`fixtures/build_pdfs.py`, gitignored PDFs) × repeats against a running API and writes `results.csv`; `analyze.py` turns it into the tables of `REPORT.md`. The runner cache-busts turn 1 per session and sends the attachment on turn 1 only, by default.
3. `evals/stress/metrics.py` defines its own `MetricResult` (there is no general evals framework in this repo yet). Facts tracked by the scenarios must not appear in the PDF filler text (a test checks it).

Embedding pipeline (Session 7), `app/embedding_pipeline/` — first step towards RAG, nothing is persisted yet (pgvector arrives in Session 8):

1. `POST /embeddings/ingest` (prefix `/embeddings`, no `/api/v1`, as the exercise specifies) takes `IngestRequest{budgets: list[Budget]}`; `sector`/`complexity` are closed `Literal`s, so unknown values are a 422.
2. `JSONStructuralChunker` (`chunker.py`): one `BudgetComponent` = one `Chunk`, no overlap or size splitting. `text` = parent header (`[Project: …]` / `[Client sector: … | Year: … | Main tech: …]`) + component detail; `metadata` holds filterable fields that are NOT embedded; `chunk_id = "{budget_id}::{component_id}"`; `token_count` via tiktoken (encoding loaded once at import).
3. `OpenAIEmbedder` (`embedder.py`) calls `text-embedding-3-small` (1536 dims) directly through the OpenAI SDK (not `LLMWrapper`: Anthropic has no embeddings). Batches of 100, backoff 1s/2s/4s on `RateLimitError` only, one `embedding_batch_done` log per batch, `PRICE_PER_MILLION_TOKENS_USD` module constant. The client and `sleep` are injected so tests use fakes.
4. `app/dependencies.py::build_embedder()` builds it from settings (`EMBEDDING_MODEL`, `max_retries=0` so the SDK does not retry on top of our backoff) and returns `None` without `OPENAI_API_KEY`; `get_embedder()` caches it and the router maps `None` or any API error to a generic 500 (detail only in the `embeddings_ingest_failed` log).
5. `scripts/compare.py` (hand-written cosine, stdlib only) produced `SANITY_CHECK.md`; `scripts/preview_chunks.py` shows chunks/tokens with no API calls; `scripts/header_effect.py` measures how much the contextual header helps. Scripts add the project root to `sys.path` so they run both with `uv run python scripts/…` and `docker compose exec estimator python scripts/…`.

Key design points future changes should respect:

- **Prompts are versioned and immutable.** Never edit a published `vN/` in place: copy it to `vN+1/` and change it there, so every logged `prompt_version` maps to the exact text used. `system.j2` includes its own version's `examples.j2` by full path.
- **Prompt text lives only in `.j2` files** — no f-string prompts in Python. Pricing assumptions (62.50 EUR/h dev, 50 EUR/h designer) are in `system.j2`.
- **Few-shot examples are data rendered per format**: `examples.j2` defines the examples once (`{% set examples = [...] %}`) and renders them in the requested `output_format`/`detail_level`, so they never contradict the format instructions. The template tests rely on this (e.g. `confidence_pct` must not appear with `narrative`). Example costs must equal the sum of their phases.
- **Router fallback needs one model group per model** (`PRIMARY_GROUP` / `FALLBACK_GROUP` linked by `fallbacks`). Putting both deployments under the same `model_name` makes LiteLLM load-balance between them. `test_router_always_picks_primary_and_falls_back_on_failure` guards this.
- **Cache key = rendered prompt + generation knobs**, so changing a form parameter or the prompt version invalidates the cache automatically. For conversations the key covers the whole `messages` array.
- **History and memory are separate.** History (`ConversationHistory`) is the sliding window of literal messages; memory (`ProjectMetadata`) is the small set of project facts injected into the system prompt every turn. Sessions are volatile by design: they live in one process, so run a single uvicorn worker, and a restart or `--reload` drops them (clients get 404).
- **Settings are a cached singleton** via `app/config.py::get_settings` (`@lru_cache`); any change to `.env` requires restarting the service (a `--reload` is not enough). At least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` must be set; the fallback needs both.
- **Logging** is `structlog`, configured in `lifespan` (`main.py`): JSON in `production`, console in dev. Use `structlog.get_logger()` rather than stdlib `logging`.

## Tests

- `tests/prompts/` — template tests: they assert on the rendered text only (no LLM, no network, milliseconds). They do not evaluate the quality of the model's answer.
- `tests/test_stress_metrics.py` / `tests/test_stress_runner.py` — Session 6 metrics and an in-process run of the stress runner with a stub wrapper (no tokens).
- `tests/embedding_pipeline/` — Session 7: chunker on the real sample, embedder with `FakeOpenAI` (`fakes.py`: batching, order, retries), endpoint via `dependency_overrides[get_embedder]`, and the cosine of `scripts/compare.py`. No tokens.
- `tests/test_session_endpoints.py` — Session 5 integration tests with `httpx.AsyncClient` + `ASGITransport` (`pytest.mark.anyio`), overriding `get_llm_wrapper` and `get_session_store`. Test PDFs/.docx are built in memory by `tests/helpers.py`.
- Endpoint and stream tests replace the wrapper with a stub via `app.dependency_overrides[get_llm_wrapper]`; wrapper and cache tests use `fakeredis`.

## Configuration

`.env` (copied from `.env.example`) drives everything via `pydantic-settings`. Notable vars:

- `PRIMARY_MODEL` / `FALLBACK_MODEL` — LiteLLM model ids for the Router (defaults `gpt-4o-mini` / `claude-haiku-4-5-20251001`).
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — at least one is required; the fallback needs both.
- `REDIS_URL`, `CACHE_TTL` — exact-match cache (compose overrides `REDIS_URL` to `redis://redis:6379`).
- `ESTIMATOR_API_BASE_URL` — where Streamlit reaches the API (compose sets `http://estimator:8000`).
- `EMBEDDING_MODEL` — embeddings model for `/embeddings/ingest` and the scripts (default `text-embedding-3-small`); embeddings always need `OPENAI_API_KEY`.
- `MAX_TURNS` — sliding-window size in user+assistant turns (default 6, current turn included).
- `APP_ENV` — `development` | `staging` | `production` (controls log renderer).

## Docker

Multi-stage Dockerfile: `builder` installs prod-only deps with `uv sync --no-install-project --no-dev`, `runtime` is a clean `python:3.11-slim` that only carries `/app/.venv`, `app/`, `ui/`, `streamlit_app.py`, `scripts/` and `data/`, runs as non-root `appuser`. There is a Docker-native HEALTHCHECK against `/health`. `docker-compose.yml` runs three services from that image plus Redis: `estimator` (API, :8000), `streamlit` (UI, :8501) and `redis`. It bind-mounts `./app` (plus `scripts/` and `data/` for the API, and `streamlit_app.py`, `ui/` for Streamlit) and adds `--reload` for dev — strip both for any production deployment.
