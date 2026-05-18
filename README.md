# automl — Agentic Price & Promo Optimisation

A 14-agent DAG that ingests a weekly SKU × store × week panel, groups SKUs
into Price-Pack Groups (PPGs), fits price-elasticity models, decomposes
drivers, optimises prices/promos under business constraints, and renders an
insights report. The orchestrator streams progress over SSE and pauses at
approval gates (PPG mapping, modelling, optimisation) for human sign-off.

Every agent is **inspectable in the UI**: the run page renders an inline
data-visibility layer next to each step — coverage heatmap, dbt + Great
Expectations results, PPG scatter (three views), per-feature histograms,
VIF bars, correlation heatmaps, plus an *Agent thinking* panel showing the
LLM prompt / response per call (or a dry-run badge when the deterministic
fallback fired).

**What's working today** (see `progress.md` for the per-phase acceptance
metrics):

- **Phase 0–1** ✅ scaffolding, ingestion (DuckDB + dbt + GE), PPG mapping
  (100% SKU agreement vs synthetic truth), PPG selection.
- **Phase 2** ✅ EDA, feature engineering, VIF + correlation refinement
  (max VIF 7.92, max |corr| 0.91, `log_price` retained).
- **Phase 2a** ✅ visible run page — every Phase 1/2 agent has at least one
  inline chart; LLM trace artefact per LLM-using agent.
- **Phase 3+** — modelling, decomposition, optimisation, validation,
  insights (planned).

See `CLAUDE.md` for contributor conventions.

## Stack

- **Python 3.11** managed by [`uv`](https://docs.astral.sh/uv/)
- **FastAPI** + SSE backend, async orchestrator
- **DuckDB** warehouse, **dbt** mart (`main.panel`), **Great Expectations** suites
- **Anthropic SDK** for per-agent LLM narratives (dry-run by default)
- **Next.js 14** + Tailwind frontend (App Router, TypeScript)
- **Apache ECharts** for the visualisation layer

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

The natural workflow is **CLI seed → API + web → click _Run pipeline_**.
You don't need the CLI to drive runs — the web UI starts one and streams
events live.

```bash
# 1. Install Python deps into the project venv
uv sync

# 2. Generate the synthetic panel (writes data/synthetic.csv)
uv run automl seed

# 3a. (Optional) Smoke test from the CLI without the UI
uv run automl run --data data/synthetic.csv --no-gates --out runs

# 3b. Or drive runs from the UI — start both services, then visit localhost:3000
uv run uvicorn api.main:app --port 8000   # in one terminal
cd web && pnpm install && pnpm dev        # in another
#   http://localhost:3000 → "Run pipeline" → /runs/<id>
```

What you'll see at `/runs/<id>` once the run is going:

- A **vertical step tracker** with all 14 agents — status pill, confidence
  chip, duration, expandable disclosure per step.
- The first six steps render their own **inline visuals** when expanded:
  - *Ingestion* — data preview, schema (column / dtype / role / null %),
    SKU × week coverage heatmap, weekly trend, dbt + GE quality panel
    (pass / warn / fail pills with rule message + row counts), anomaly
    list with severity.
  - *PPG Mapping* — three SKU scatters (Tier × log-price, behaviour-based
    log-units × elasticity proxy, faceted brand × pack by category) inside
    an inline tab strip, plus a within-PPG price-box plot and the full
    PPG → SKU breakdown table.
  - *PPG Selection* — per-PPG stacked eligibility bars with the 0.60
    threshold line.
  - *EDA* — weekly trend, pairwise correlation heatmap of numeric
    candidates, ranked target-relationship table, EDA findings.
  - *Feature Engineering* — 16-tile histogram grid (one per engineered
    column) with μ / σ / n.
  - *Feature Refine* — VIF bar with threshold marker (red ≥ threshold,
    amber ≥ 70 %, emerald below), refined-set correlation heatmap, drop
    log with reasons, kept-feature pills.
- An **Agent thinking** sub-section under every LLM-using agent —
  collapsible per-call panes showing the system prompt, user prompt, and
  raw response, with a *dry-run* badge or live `tokens_in ↓ / tokens_out ↑
  / $cost` row.
- An **Artifact gallery** at the bottom grouping every JSON / CSV /
  parquet the run produced, with download links.

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

Per-run artefacts land in `runs/<run_id>/`. The file set has grown with
Phase 2a — every agent now writes both its narrative artefact and one or
more chart-ready JSONs the frontend renders directly.

| File | Produced by | Used for |
|---|---|---|
| `state.json` | orchestrator | live agent statuses, costs |
| `events.jsonl` | event bus | one JSON event per line (SSE replay) |
| `warehouse.duckdb` | dbt build | `main.panel` mart |
| `ingestion_report.json`, `data_profile.json`, `ingestion_findings.json` | ingestion | dbt + GE results, column profile, LLM anomaly narrative |
| `coverage_grid.json`, `weekly_trend.json`, `quality_results.json` | ingestion (charts) | SKU × week coverage, weekly aggregate, normalised quality list |
| `ppg_mapping.json`, `ppg_mapping_table.json` | ppg_mapping | full assignment blob + flat table |
| `ppg_scatter_tier.json`, `ppg_scatter_behaviour.json`, `ppg_scatter_facet.json`, `ppg_price_box.json` | ppg_mapping (charts) | three scatter views + per-PPG price quantiles |
| `ppg_selection.json`, `ppg_eligibility_bars.json` | ppg_selection | scored eligibility + stacked-bar breakdown |
| `feature_candidates.json` | feature_selection | candidate columns + role tagging |
| `eda_report.json`, `eda_corr_matrix.json` | eda | overview / numeric summary / target-relationship / pairwise corr + heatmap-shaped matrix |
| `features.parquet` (or `.csv`), `feature_engineering.json`, `feature_histograms.json` | feature_engineering | engineered frame + summary + per-column histograms |
| `feature_refine.json`, `corr_refined.json` | feature_refine | kept / dropped / VIF + refined-set correlation matrix |
| `<agent>_llm_trace.json` | every LLM-using agent | system / user / response / model / tokens / dry-run flag, one per call (disable with `LLM_TRACE=false`) |

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
- Streams agent events over SSE and renders a vertical step tracker with a
  progress bar, current-step badge, and elapsed-time counter.
- Per agent, when the disclosure is open: reasoning, ordered tool calls,
  the *Agent thinking* panel (system / user / response per LLM call), the
  per-agent inline charts and tables described in **Quickstart**, and a
  link to every artefact the agent produced.
- Approve / reject controls release the active approval gate without
  leaving the run page.
- An artifact gallery at the bottom of the page indexes every JSON / CSV /
  parquet produced by the run.

Throwaway design-record pages live under `web/app/dev/` (`/dev/inline`,
`/dev/tabs`, `/dev/subroutes/*`) — these are the layout mockups that
informed the inline-visuals choice; they're not part of the production
flow.

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
| `LLM_TRACE` | `true` | Set to `false` to suppress the per-agent `<agent>_llm_trace.json` audit artefact. Dry-run runs still capture a trace with `dry_run: true` by default — disable here when the prompts may carry sensitive row samples and the run dir will be shared. |
| `LLM_DRY_RUN` | _(unset)_ | Force every LLM call into the deterministic fallback even when `ANTHROPIC_API_KEY` is set. Useful for cheap CI runs. |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Frontend → backend base URL. Inlined at **build time** for the Next.js bundle. |

## Tests

`pytest` is not in the project's runtime deps to keep the install lean;
pull it in on the fly with `uv`:

```bash
uv run --with pytest pytest tests/ -q
# 22 passed
```

The suite covers:

- **PPG clustering** (`test_ppg_clustering.py`) — synthetic panel through
  dbt + the clusterer; asserts ≥ 95 % SKU agreement vs `synthetic/truth.json`
  (currently 100 %, 48 / 48).
- **Feature refine** (`test_feature_refine.py`) — end-to-end VIF +
  correlation pruning; asserts `max VIF < 10`, `max |corr| ≤ 0.95`,
  `log_price` retained.
- **Chart-data builders** (`test_charts.py`) — every builder in
  `core/data/charts.py` against a dbt-built synthetic warehouse + the
  graceful-degradation case where `brand` / `category` / `pack_size` are
  absent.
- **LLM trace** (`test_llm_trace.py`) — base `Agent.call_llm` captures
  system / user / response into `<agent>_llm_trace.json`; `LLM_TRACE=false`
  cleanly suppresses; failures-after-LLM still flush before re-raising.
- **Gitignore guard** (`test_gitignore_sources.py`) — fails fast if any
  source path under `core/`, `api/`, `cli/`, `synthetic/`, `tests/`,
  `web/app/`, `web/components/`, `web/lib/` is silently matched by a
  `.gitignore` rule (catches the unanchored `runs/` trap).

## Repository layout

```
api/                   FastAPI app + routes (runs, events, uploads, artefacts, approvals)
cli/                   Typer CLI (`automl run|seed|baseline-create`)
core/
  agents/              One file per agent; all inherit core.agents.base.Agent
  data/                Ingestion, dbt runner, GE runner, profiling tools,
                       chart-ready data builders (charts.py), ingestion report
  features/            EDA tools, engineering pipeline, VIF + correlation refine
  llm/                 AnthropicClient + per-agent model routing + LLM trace
  orchestrator/        RunState, EventBus, gates, async runner
  ppg/                 Per-SKU features, clustering, scoring
dbt/automl_dbt/        dbt project (DuckDB profile, staging + panel mart, tests)
synthetic/             Synthetic data generator + ground-truth JSON
tests/unit/            pytest suites (PPG, feature refine, charts, LLM trace, ...)
web/
  app/                 Next.js routes (`/`, `/runs`, `/runs/[id]`, `/dev/*` mockups)
  components/
    charts/            ECharts wrappers (CoverageHeatmap, TrendChart,
                       PPGScatter, PPGPriceBox, EligibilityBars,
                       CorrHeatmap, VIFBar, FeatureHistograms)
    tables/            DataPreview, SchemaTable, QualityPanel, AnomalyTable,
                       DropLog, TargetRelationship
    AgentCard.tsx      Per-agent disclosure tile
    AgentVisuals.tsx   Per-agent inline chart container
    AgentThinking.tsx  Collapsible LLM-trace panel (system / user / response)
    PPGTabs.tsx        Inline tab strip used inside the PPG mapping card
    PPGTable.tsx       PPG → SKU breakdown with approve / reject controls
  lib/                 API client, SSE hooks, types, agent metadata
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

### `uv sync` fails with an SSL / certificate error

This is almost always a corporate proxy that re-signs TLS traffic with a
private CA. Pick one:

```bash
# Option A — use the OS / system trust store (preferred on machines that
# already trust the corporate CA at the OS level).
export UV_NATIVE_TLS=1
uv sync

# Option B — point uv at a specific CA bundle (e.g. one shipped by IT).
export SSL_CERT_FILE=/path/to/corp-ca-bundle.crt
uv sync

# Option C — last resort, allow uv to skip TLS verification for the
# package mirrors only. Don't leave this in your shell rc.
export UV_INSECURE_HOST="pypi.org files.pythonhosted.org"
uv sync
```

### `dbt` complains about missing `dbt_utils` / `dbt_expectations` macros

The dbt mart depends on two external packages (`packages.yml`). The runner
auto-installs them the first time it builds (it runs `dbt deps` when
`dbt/automl_dbt/dbt_packages/` is missing), so a plain
`uv run automl run --data data/synthetic.csv` is enough on a fresh checkout.
If you want to install them manually:

```bash
uv run dbt deps --project-dir dbt/automl_dbt --profiles-dir dbt/automl_dbt
```

### Other

- **`automl: command not found`** — you skipped `uv sync` or aren't using
  `uv run`. Always prefix Python entry points with `uv run`.
- **dbt profile not found** — the repo's `dbt/automl_dbt/profiles.yml` is
  self-contained and the runner sets `DBT_PROFILES_DIR` per run; a stale
  `~/.dbt/profiles.yml` can still be picked up by a manual `dbt` invocation
  — pass `--profiles-dir dbt/automl_dbt` to override it.
- **SSE stream stalls behind a reverse proxy** — disable response buffering
  on the `/runs/*/events` path (nginx: `proxy_buffering off`).
- **LLM output looks templated** — `ANTHROPIC_API_KEY` is unset, so every
  agent is running its dry-run fallback. This is intentional and expected
  for cheap CI / smoke runs.
