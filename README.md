# automl — Agentic Price & Promo Optimisation

A 14-agent DAG that ingests a weekly SKU × store × week panel, groups SKUs
into Price-Pack Groups (PPGs), fits price-elasticity models, decomposes
drivers, optimises prices/promos under business constraints, and renders an
insights report. The orchestrator streams progress over SSE and pauses at
approval gates (PPG mapping, modelling, optimisation) for human sign-off.

See `progress.md` for the phased plan and `CLAUDE.md` for contributor
conventions.

## Stack

- **Python 3.11** managed by [`uv`](https://docs.astral.sh/uv/)
- **FastAPI** + SSE backend, async orchestrator
- **DuckDB** warehouse, **dbt** mart (`main.panel`), **Great Expectations** suites
- **Anthropic SDK** for per-agent LLM narratives (dry-run by default)
- **Next.js 14** + Tailwind frontend (App Router, TypeScript)

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | runtime |
| `uv` | latest | dependency + venv management |
| Node | 20+ | Next.js frontend |
| `pnpm` | 9+ | frontend package manager |
| `git` | any | version control |

Install `uv` (one-liner):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quickstart (60 seconds)

```bash
# 1. Install Python deps into the project venv
uv sync

# 2. Generate the synthetic panel (writes data/synthetic.csv)
uv run automl seed

# 3. Run the full pipeline end-to-end (gates disabled for a smoke run)
uv run automl run --data data/synthetic.csv --no-gates --out runs

# 4. Inspect outputs
ls runs/<run_id>/
#   state.json   events.jsonl   warehouse.duckdb
#   ingestion_findings.json   ppg_mapping_table.json   ppg_selection.json   ...
```

## CLI

The `automl` console script is registered in `pyproject.toml`. All commands
go through `uv run` so the project venv and lockfile are respected.

```bash
# End-to-end pipeline with approval gates enabled
#   (orchestrator pauses after gated agents; resume via POST /approve)
uv run automl run --data data/synthetic.csv --out runs

# Same, but skip gates for non-interactive smoke runs
uv run automl run --data data/synthetic.csv --no-gates --out runs

# Regenerate the synthetic panel from synthetic/truth.json
uv run automl seed

# Capture a distribution snapshot from a completed run for drift checks
uv run automl baseline-create runs/<run_id> --name <name>
```

Per-run artefacts land in `runs/<run_id>/`:

| File | Produced by |
|---|---|
| `state.json` | orchestrator (live agent statuses, costs) |
| `events.jsonl` | event bus (one JSON event per line) |
| `warehouse.duckdb` | dbt build (`main.panel` mart) |
| `ingestion_report.json` | ingestion agent |
| `ppg_mapping_table.json`, `ppg_mapping.json` | ppg_mapping agent |
| `ppg_selection.json` | ppg_selection agent |
| `data_profile.json` | ingestion profiling tools |

## API

The FastAPI app exposes a small REST surface plus an SSE stream.

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
# health check:  curl http://localhost:8000/health
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/runs` | Create a new run (multipart upload or `data_path`) |
| `GET` | `/runs` | List runs |
| `GET` | `/runs/{id}` | Run state + agent statuses |
| `GET` | `/runs/{id}/events` | SSE stream of orchestrator events |
| `POST` | `/runs/{id}/approve?agent=<name>` | Release an approval gate |
| `POST` | `/runs/{id}/reject?agent=<name>` | Reject and halt the run |
| `POST` | `/uploads` | Upload a CSV (returns a path usable by `/runs`) |
| `GET` | `/artifacts/{run_id}/{path}` | Read any artefact under `runs/<id>/` |

## Web UI

```bash
cd web
pnpm install
pnpm dev          # http://localhost:3000
# or for a production build
pnpm build && pnpm start
```

The UI:

- Lists runs with live status pulled from the API.
- Streams agent events over SSE and renders a timeline.
- Renders the PPG mapping with confidence badges, per-group LLM rationale,
  and eligibility scores. Approve / reject buttons release the approval
  gate on the backend.

Set `NEXT_PUBLIC_API_BASE` if the backend isn't on `http://localhost:8000`:

```bash
NEXT_PUBLIC_API_BASE=https://api.example.com pnpm build
```

## Configuration

The pipeline runs without any environment variables in dry-run mode (LLM
calls fall back to deterministic stubs that produce the same JSON shape).
For real LLM narratives:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(unset)_ | When unset, every agent uses its dry-run fallback. Set it to enable real Claude calls. |
| `ANTHROPIC_MODEL` | per-agent default in `core/llm/routing.py` | Override the model used by every agent (Opus / Sonnet / Haiku). |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Frontend → backend base URL. |

## Tests

`pytest` is not in the project's runtime deps to keep the install lean; pull
it in on the fly with `uv`:

```bash
uv run --with pytest pytest tests/ -q
```

The verification suite drives the synthetic panel through dbt and the
PPG-clustering algorithm and asserts ≥95% SKU agreement with the embedded
ground truth (`synthetic/truth.json`).

## Repository layout

```
api/                   FastAPI app + routes
cli/                   Typer CLI (`automl run|seed|baseline-create`)
core/
  agents/              One file per agent; all inherit core.agents.base.Agent
  data/                Ingestion, dbt runner, GE runner, profiling tools
  llm/                 AnthropicClient + per-agent model routing
  orchestrator/        RunState, EventBus, gates, async runner
  ppg/                 Per-SKU features, clustering, scoring
dbt/automl_dbt/        dbt project (DuckDB profile, staging + panel mart)
synthetic/             Synthetic data generator + ground-truth JSON
tests/unit/            pytest suites
web/                   Next.js frontend
runs/                  Per-run artefacts (created at runtime; gitignored)
```

## Deployment

The project is two services plus a filesystem of run artefacts.

### Backend (FastAPI)

The simplest deployment is one container that builds the project venv with
`uv` and runs uvicorn:

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Operational notes:

- `runs/` must be a writable, persistent volume — every run writes a DuckDB
  warehouse and several JSON artefacts here, and the API serves files from
  it.
- A single uvicorn process is fine for demos. For real workloads, front
  uvicorn with a reverse proxy that supports SSE buffering disabled (nginx
  with `proxy_buffering off` for `/runs/*/events`).
- Set `ANTHROPIC_API_KEY` as a secret to enable real LLM calls. Without
  it, the pipeline still runs end-to-end on deterministic fallbacks.
- CORS is currently open (`allow_origins=["*"]`) — lock this down in
  `api/main.py` before exposing publicly.

### Frontend (Next.js)

```dockerfile
FROM node:20-alpine AS deps
WORKDIR /app
COPY web/package.json web/pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile

FROM node:20-alpine AS build
WORKDIR /app
COPY web/ ./
COPY --from=deps /app/node_modules ./node_modules
ARG NEXT_PUBLIC_API_BASE
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
RUN corepack enable && pnpm build

FROM node:20-alpine
WORKDIR /app
COPY --from=build /app ./
EXPOSE 3000
CMD ["node_modules/.bin/next", "start", "-p", "3000"]
```

Pass `NEXT_PUBLIC_API_BASE` at **build time** (it's inlined into the static
bundle), not at runtime.

### docker-compose sketch

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
    volumes:
      - ./runs:/app/runs
      - ./data:/app/data
  web:
    build:
      context: .
      dockerfile: web/Dockerfile
      args:
        NEXT_PUBLIC_API_BASE: http://localhost:8000
    ports: ["3000:3000"]
    depends_on: [api]
```

## Troubleshooting

- **`automl: command not found`** — you skipped `uv sync` or aren't using
  `uv run`. Always prefix with `uv run`.
- **dbt fails on first run** — dbt-duckdb needs `~/.dbt/profiles.yml` to be
  absent or to point at the project profile. The repo's `dbt/automl_dbt/`
  is self-contained; the runner sets `DBT_PROFILES_DIR` per run.
- **SSE stream stalls behind a proxy** — disable response buffering on the
  `/runs/*/events` path (nginx: `proxy_buffering off`).
- **LLM output looks templated** — `ANTHROPIC_API_KEY` is unset, so every
  agent is running its dry-run fallback. This is intentional and expected
  for cheap CI / smoke runs.
