# CLAUDE.md

Project-level guide for Claude Code agents working on this repo.

## What this is

An agentic price-and-promo optimisation platform for CPG. A weekly panel (SKU
× store × week) goes through a 14-agent DAG that ingests + validates the data,
groups SKUs into Price-Pack Groups, builds features, fits price-elasticity
models, decomposes drivers, simulates scenarios, optimises prices/promos under
constraints, validates, and writes an insights report.

The build ships in seven incremental phases (see `progress.md`). Each phase
must ship a runnable end-to-end slice so the system stays demoable.

## Tech stack

- **Python 3.11**, deps managed by `uv` (`pyproject.toml`, `uv.lock`).
- **Backend:** FastAPI + SSE, async orchestrator, Pydantic state.
- **Data:** DuckDB warehouse, dbt for the canonical mart and schema/singular
  tests, Great Expectations 1.x for distribution/relationship suites.
- **ML/Stats:** pandas, numpy, scikit-learn, scipy. Later phases add
  statsmodels, lightgbm + shap, pymc, pulp.
- **LLM:** `anthropic` SDK via `core/llm/client.py`, with prompt-caching
  hints and a `dry_run` mode (default when `ANTHROPIC_API_KEY` is missing).
- **Frontend:** Next.js 14 (App Router, TypeScript), Tailwind, pnpm.
- **Synthetic data:** `synthetic/generator.py` builds a panel with embedded
  ground-truth elasticities; `synthetic/truth.json` is the verification anchor.

## Repo layout

```
api/                   FastAPI app + routes (runs, events SSE, uploads,
                       artifacts, approvals)
cli/                   Typer CLI - `automl run|seed|baseline-create`
core/
  agents/              One file per agent. All inherit core.agents.base.Agent.
                       StubAgent fills stages that haven't been implemented yet.
  data/                io, schema, dbt_runner, ge_runner, expectations,
                       ingestion_report, profiling tools.
  llm/                 AnthropicClient + per-agent model routing.
  orchestrator/        RunState, AgentResult, EventBus, gates, async runner.
  ppg/                 Per-SKU feature aggregation, clustering, scoring.
dbt/automl_dbt/        dbt project (DuckDB profile, staging + panel mart,
                       schema + singular tests, dbt_utils + dbt_expectations).
data/                  Synthetic CSV + .gitkeep.
synthetic/             Generator + ground-truth JSON.
tests/unit/            pytest (sync + async via pytest-asyncio).
web/                   Next.js shell.
runs/                  Per-run artifacts: state.json, events.jsonl, the
                       DuckDB warehouse, agent-produced JSON.
```

## Key commands

Always invoke Python through `uv run` so the project venv and lockfile are
respected:

```bash
# Generate the synthetic panel into data/synthetic.csv
uv run automl seed

# End-to-end pipeline (gates skipped for smoke runs)
uv run automl run --data data/synthetic.csv --no-gates --out runs

# With approval gates active (orchestrator pauses; POST /approve to resume)
uv run automl run --data data/synthetic.csv --out runs

# Capture a baseline distribution snapshot for future drift checks
uv run automl baseline-create runs/<run_id> --name <name>

# FastAPI dev server
uv run uvicorn api.main:app --port 8000

# Tests (pytest is not in the project deps; use uv's --with)
uv run --with pytest pytest tests/ -q

# Next.js
cd web && pnpm install && pnpm dev    # or pnpm build
```

## Conventions

### Agents

- One agent per file under `core/agents/`. Subclass `core.agents.base.Agent`,
  set `name`, implement `async def _execute(self, run, result)`.
- The base class handles lifecycle (`agent_started` / `agent_finished` /
  `agent_failed`), error trapping, and token/cost accounting.
- Emit progress with `await self.emit(run, "tool_called", {...})`.
- Persist outputs as JSON artifacts in `Path(run.run_dir)` and append
  `ArtifactRef` to `result.artifacts`. The artifact endpoint serves files
  from `runs/<run_id>/` via path-traversal-blocked `FileResponse`.
- Set `result.outputs` to a small JSON-serialisable summary (cards/UI read
  this), `result.reasoning` to a short narrative, and `result.confidence`
  to a 0-1 number.

### LLM use

- Every agent that calls the LLM also has a **deterministic dry-run fallback**
  that produces the same JSON shape. The pipeline must run end-to-end without
  an API key — that's how CI and local smoke tests stay cheap.
- Prefer `self.call_llm(result, system=..., user=..., max_tokens=...)`.
  Model routing is per-agent in `core/llm/routing.py`.
- Parse LLM output with `json.loads` inside `try/except` and fall back on
  the deterministic dry-run when the model returns non-JSON.
- `AnthropicClient` supports four providers, picked automatically by env:
  `dry_run` (default when no creds), `api` (`ANTHROPIC_API_KEY`),
  `oauth` (`ANTHROPIC_AUTH_TOKEN` from `claude setup-token` etc.), and
  `cli` (shells out to the local `claude` binary). `cli` is never picked
  implicitly — you must set `LLM_PROVIDER=cli`. Integration tests use the
  `live_llm` marker and only hit the network when `RUN_LIVE_LLM=true`.

### Orchestration

- `AGENT_ORDER` (in `core/orchestrator/state.py`) is the canonical DAG order.
- Default approval gates: `ppg_mapping`, `modeling`, `optimization`. The
  runner pauses after the gated agent and waits on a `GateRegistry` event
  released by `POST /runs/{id}/approve` or `/reject`.
- Disable gates per-run with `gates_enabled=False` (`--no-gates` on the CLI).

### Data

- The canonical mart is `main.panel` in the per-run DuckDB at
  `runs/<id>/warehouse.duckdb`. Always read through dbt's mart, never the raw
  load.
- dbt source is overridden per-run via env vars so each run gets a fresh
  warehouse — see `core/data/dbt_runner.py`.
- New data quality checks: prefer dbt schema tests or dbt-expectations for
  per-row / per-column rules, and the in-code GE suites in
  `core/data/expectations.py` for distribution/relationship rules.

### Frontend

- Server components by default; mark `"use client"` only on the file that
  actually needs hooks/state.
- API access goes through `web/lib/api.ts`. Browser-side fetches use
  same-origin `/api/*` URLs and rely on the rewrite in
  `web/next.config.mjs` to proxy to the API server (`API_PROXY_TARGET`
  or `NEXT_PUBLIC_API_BASE`, default `http://localhost:8000`). The
  server-side caller (`listRuns` in `app/runs/page.tsx`) uses the
  absolute URL from `process.env`. Don't introduce new direct
  `http://localhost:...` fetches in client code — the bundle moves
  between machines, the env doesn't.
- Per-agent UI gets re-fetched when its `agent_finished` event arrives -
  see `PPGTable.tsx` for the pattern.

### Branches + commits

- Develop on the branch named in the session brief (currently
  `claude/agentic-pricing-solution-d5Tw9`). Don't push elsewhere without
  explicit permission.
- Phase commits get a short body summarising backend / frontend / tests /
  verification, with concrete file paths. Don't include marketing copy.
- Update `progress.md` when a phase flips state. Don't pre-mark future
  phases.

## Verification gates per phase

Each phase has a concrete acceptance metric in `progress.md`. Don't mark a
phase complete until that metric is verified end-to-end:

- **P1:** PPG mapping >=95% SKU agreement vs synthetic truth.
- **P2:** Final feature set has VIF<10 and no |corr|>0.95 pairs.
- **P3:** Log-log on synthetic recovers truth elasticity sign for >=7/8 PPGs.
- **P4:** Decomposition reconciles to observed units within tolerance.
- **P5:** Optimisation respects ladder / margin floor / comp-gap constraints
  and holdout WAPE is reported.
- **P6:** HTML + PDF report renders cleanly; cost dashboard sums per-agent
  tokens/$.

## House style

- **No unrequested files.** Don't create planning, decision, analysis, or
  README files unless the user asks for one.
- **No defensive code beyond boundaries.** Trust internal invariants. Only
  validate user input, external APIs, and tool-result shapes.
- **No comments that restate the code.** Comments are for the WHY when it
  isn't obvious — invariants, workarounds, surprising behaviour. Never
  reference the current task or recent commits in a comment.
- **No backwards-compat shims** unless the user asks. If something is unused,
  delete it.
- **Smallest commit that works.** Don't pile cleanup onto a feature commit.
- **dry-run by default** on every new LLM call. Real spend is opt-in.
- **Anchor every `.gitignore` pattern that names a top-level dir.** Use
  `/runs/`, not `runs/`. An unanchored pattern matches anywhere in the tree
  and will silently swallow real source files (this trap once cost us the
  Next.js `web/app/runs/` route segment — commits looked clean, screenshots
  worked, the route only 404'd for anyone cloning from GitHub). The
  `test_gitignore_does_not_swallow_sources` pytest in `tests/unit/`
  enforces this — keep it green.
- **Never compare paths with string ops; use `pathlib`.** Code like
  `str(target).startswith(str(base) + "/")` works on POSIX and silently
  breaks on Windows because resolved paths use `\`, not `/` — so every
  legitimate request looks like a path-traversal attempt. Use
  `target.relative_to(base)` (raises `ValueError` if outside) or
  `target.is_relative_to(base)` (3.9+). This trap once made every artifact
  fetch return 400 on Windows while the same code worked on Linux CI. The
  endpoint check lives at `api/routes/artifacts.py`; regression coverage
  is `tests/unit/test_artifacts_route.py`.
- **Bake nothing host-specific into the Next.js client bundle.** Don't
  read `process.env.NEXT_PUBLIC_*` for hostnames the browser will fetch
  from — those values are inlined at build time and won't survive moving
  the build to another machine. The browser always hits same-origin
  `/api/*`; `web/next.config.mjs` proxies that prefix to the API server
  (configurable at next-server startup via `API_PROXY_TARGET`). The one
  exception is the SSR caller in `web/app/runs/page.tsx`, which uses an
  absolute URL because the server process can resolve `localhost:8000`
  directly. When adding a new fetch, route it through `web/lib/api.ts`
  so it picks up the right base automatically.

## When in doubt

Read `progress.md` to know the current phase and acceptance metric. Read
`core/orchestrator/state.py` for the shared `RunState` / `AgentResult` /
`ArtifactRef` contracts that every agent both reads and writes. Read
`core/agents/base.py` for the agent lifecycle.
