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
- **Phase 2a** ✅ visible run page — every agent has at least one
  inline chart; LLM trace artefact per LLM-using agent.
- **Phase 3** ✅ modelling (log-log + semi-log sign-retry + LightGBM,
  ranked by hold-out WAPE), SHAP attribution, empirical-Bayes hierarchical
  shrinkage with forest plot.
- **Phase 4** ✅ closed-form decomposition for OLS winners, ablation-based
  decomposition for LightGBM, price × promo simulation grid; reconciles to
  predicted within 1e-6 across all 8 synthetic PPGs.
- **Phase 5** ✅ scipy continuous warm-start → PuLP MILP under ladder /
  margin floor / comp gap / move guardrail, soft-relax fallback with
  binding-violation surface, edit-and-re-solve gate loop, rolling-origin
  CV validation with sign-stability + WAPE + ε-CV verdicts.
- **Phase 6** ✅ insights agent with HTML + WeasyPrint PDF report, cost
  dashboard, executive banner, run-replay scrubber with keyboard shortcuts,
  cross-run sidebar, fitted-vs-actual + decomposition + simulation +
  constraint-binding + residual-histogram charts.
- **Modularity & config refactor** ✅ central `core/config.py`
  (pydantic-settings) with per-agent `MODEL_<AGENT>` overrides; frontend
  theme / api-config / chart-config / agent-meta consolidated.
- **Per-stage FAQ + corner cases** ✅ every agent card carries a
  collapsible *Common questions & corner cases* disclosure; full reference
  in `docs/stage-faqs.md`.
- **Real CPG scanner data + published-elasticity benchmarks** ✅
  Dominick's Finer Foods loader (`automl prepare-dominicks`) lands the
  Kilts Center panel in the canonical schema so the pipeline runs on real
  data, not just the synthetic generator. The validation agent now scores
  each PPG's recovered elasticity against Hoch et al. (1995) Dominick's
  category ranges, with the Bijmolt et al. (2005) meta-analysis grand
  mean (−2.62) as the fallback for categories Hoch didn't cover.

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
- Each agent's card surfaces its **inline visuals** when expanded:
  - *Ingestion* — data preview, schema, SKU × week coverage heatmap, weekly
    trend, dbt + GE quality panel (pass / warn / fail pills), anomaly list.
  - *PPG Mapping* — three SKU scatters (Tier × log-price, behaviour-based,
    faceted brand × pack by category) inside an inline tab strip, price-box
    plot, full PPG → SKU breakdown.
  - *PPG Selection* — per-PPG stacked eligibility bars with the 0.60
    threshold line.
  - *EDA* — weekly trend, pairwise correlation heatmap, ranked
    target-relationship table.
  - *Feature Engineering* — 16-tile histogram grid (μ / σ / n).
  - *Feature Refine* — VIF bar, refined-set correlation heatmap, drop log,
    kept-feature pills.
  - *Modeling* — candidates table (winner highlighted, attempts expandable),
    mean-|SHAP| bar, elasticity forest plot (OLS vs EB-shrunk posterior),
    fitted-vs-actual scatter with log-space `r`.
  - *Decomposition* — stacked area of base + due-by-group per week with the
    observed line overlaid, per-PPG selector.
  - *Simulation* — price-multiplier × promo heatmap toggleable between
    revenue / margin / units.
  - *Optimization* — recommendation table with %Δ chips, constraint-binding
    bar showing per-PPG slacks, inline **constraint editor** that drives
    the rerun gate loop.
  - *Validation* — verdict table (pass / warn / fail per PPG), residual
    histogram of pooled hold-out residuals.
  - *Insights* — exec headline + KPI tiles + HTML/PDF download buttons,
    per-PPG recommendations.
- A *Common questions & corner cases* disclosure per card with the FAQ for
  that stage (full reference in `docs/stage-faqs.md`).
- An **Agent thinking** sub-section under every LLM-using agent —
  collapsible per-call panes showing the system prompt, user prompt, and
  raw response, with a *dry-run* badge or live `tokens_in ↓ / tokens_out ↑
  / $cost` row.
- An **Executive banner** at the top once insights finishes; a **Replay
  scrubber** above the timeline (Space = play/pause, ←/→ = step); a
  **Cost dashboard** + **Artifact gallery** at the bottom.

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

# Convert a downloaded Dominick's archive into a panel-shaped CSV
#   (see "Using real Dominick's data" below for the prerequisite)
uv run automl prepare-dominicks --categories yogurt,beer --out data/dominicks.csv

# Capture a distribution snapshot from a completed run for drift checks
uv run automl baseline-create runs/<run_id> --name <name>
```

### Using real Dominick's data

The pipeline also accepts the Dominick's Finer Foods scanner panel published
by the [Kilts Center, University of Chicago Booth](https://www.chicagobooth.edu/research/kilts).
The license forbids redistribution, so the raw files have to be
downloaded after signing the Kilts data-use agreement — they're not in
this repo.

1. Sign the Kilts agreement and download the per-category archives
   (movement file `w<code>.csv` + UPC dictionary `upc<code>.csv` per
   category, e.g. `wyog.csv` + `upcyog.csv` for yogurt).
2. Drop them anywhere under `data/dominicks-raw/` (gitignored). Nested
   subdirectories are fine — the loader globs by filename.
3. Run the adapter:

   ```bash
   uv run automl prepare-dominicks \
       --raw-dir data/dominicks-raw \
       --categories yogurt,beer,soft_drinks \
       --out data/dominicks.csv
   ```

   Use `--categories all` for every known Dominick's category. The loader
   maps each category's movement + UPC dictionary onto the canonical
   panel schema (per-unit `price = PRICE/QTY`, trailing-13-week non-promo
   `base_price`, `SALE ∈ {B,S,C}` → `tpr_flag`, week 1 anchored to
   1989-09-14, US-holiday weeks tagged via the `holidays` package).

4. Feed the resulting CSV straight into the pipeline:

   ```bash
   uv run automl run --data data/dominicks.csv --no-gates --out runs
   ```

The full mapping lives at `core/data/loaders/dominicks.py`; the category
code → benchmark-key table is at `core/data/loaders/dominicks_categories.py`.
The validation agent automatically joins each PPG's recovered elasticity
to the published category band (Hoch 1995 / Bijmolt 2005) — see *Published
elasticity benchmarks* below.

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
| `modeling_results.json`, `elasticity_per_ppg.json`, `shap_per_ppg.json`, `hierarchical_posterior.json`, `fitted_vs_actual.json` | modeling | per-PPG fits + winner pick + SHAP + EB-shrunk posterior + fitted-vs-actual scatter |
| `results_reasoning.json`, `model_choice_summary.json` | results_reasoning | per-PPG pass/warn/fail verdict + flat row-shape summary |
| `decomposition_per_ppg_week.json`, `decomposition_summary.json`, `decomposition_table.json` | decomposition | weekly base + due-by-group + residual; closed-form for OLS, ablation for LightGBM |
| `simulation_grid.json`, `simulation_summary.json`, `simulation_table.json` | simulation | price × promo sweep + revenue / margin-optimal cells |
| `optimization_results.json`, `optimization_table.json`, `optimization_constraints.json` | optimization | continuous + MILP solution per PPG, resolved constraints, `chosen_slacks` per cell |
| `validation_report.json`, `validation_table.json`, `validation_residuals.json` | validation | rolling-origin CV verdict + per-fold detail + pooled hold-out residuals + per-PPG `benchmark_status` / `benchmark_low` / `benchmark_high` / `benchmark_source` (Hoch 1995 / Bijmolt 2005) |
| `insights_summary.json`, `cost_summary.json`, `report.html`, `report.pdf` | insights | exec headline + KPIs + recommendations table, per-agent token/cost rollup, HTML + (WeasyPrint) PDF report |
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
| `POST` | `/runs/{id}/rerun?agent=<name>` | Edit-and-re-solve loop for `RERUNNABLE_AGENTS` (currently `optimization`). JSON body merges into `run.options[agent]` before re-execution. |
| `POST` | `/uploads` | Upload a CSV (returns a path usable by `/runs`) |
| `GET` | `/artifacts/{run_id}/{path}` | Read any artefact under `runs/<id>/` |

## Frontends

There are two Next.js applications. Both read run artifacts from the API; neither carries a hostname in the browser bundle — all fetches go through same-origin `/api/*` and the Next.js proxy injects the bearer token server-side.

### Operator UI (`web/` — port 3000)

The operator frontend controls pipeline runs and monitors agent progress in real time.

```bash
cd web
pnpm install
pnpm dev          # http://localhost:3000
# or for a production build
pnpm build && pnpm start
```

**Pages**

| Route | Description |
|---|---|
| `/` | Home — start a new run (upload CSV or server-side path), see recent runs |
| `/runs` | All runs — active and archived; archive / restore / delete |
| `/runs/[id]` | Run detail — agent timeline, cost dashboard, artifact gallery |
| `/runs/[id]/eda` | Advanced EDA dashboard for the run |

**Starting a run (`/`):**

Pick a data source, toggle options, click **Run pipeline**:

- **Upload CSV** — local file with columns `sku, store_id, week_start, units, price, tpr_flag` (max 200 MiB).
- **Server-side path** — path already present on the API server (default: `data/synthetic.csv`).

Options:

| Option | Default | Effect |
|---|---|---|
| Agent mode | on | LLM narratives and summaries. Off → deterministic fallbacks, zero spend. |
| Approval gates | off | Pause after PPG mapping, modeling, and optimization for manual review. |
| Pick modelling grain after ingestion | on | Pause after PPG mapping to choose modelling granularity (chain × PPG, store × brand, etc.). |

**Run detail (`/runs/[id]`):**

The timeline shows all 14 agents in DAG order. Each agent card expands to show: live event stream, inline charts and tables, reasoning, *Agent thinking* pane (LLM prompt + response), artifact links, and a *Common questions & corner cases* FAQ.

Additional panels:

- **Executive banner** — headline insight and PDF download once `insights` completes.
- **Replay bar** — scrub through SSE events to replay the run at any timestamp (`Space` play/pause, `←`/`→` step one event).
- **Cost dashboard** — per-agent token counts and estimated USD spend.
- **Artifact gallery** — direct links to every JSON / CSV / Parquet artifact.
- **Sidebar** — lists all runs with status dots; click any to switch without a full reload.

**Runs list (`/runs`):**

Toggle between Active and Archived. Per row: click the ID to open the detail page; **archive** to hide from the active list (run must be finished); **restore** to move back; **delete** to permanently remove from disk.

If the API server is not on `http://localhost:8000`, point the proxy at it at **runtime** (no rebuild needed):

```bash
API_PROXY_TARGET=https://api.example.com API_AUTH_TOKEN=$TOKEN pnpm start
```

Throwaway design-record pages live under `web/app/dev/` — these are layout mockups that informed the inline-visuals choice; they are not part of the production flow.

---

### Business UI (`web-client/` — port 3001)

A presentation-ready view aimed at category managers and CPG operators who want decisions, not diagnostics. It reads artifacts from a completed run via `?runId=<id>`.

```bash
cd web-client
pnpm install
pnpm dev          # http://localhost:3001
```

Reach it from the operator UI via the **Business view ↗** link in the run subnav.

**Pages**

| Route | Description |
|---|---|
| `/dashboard?runId=` | Executive summary — headline KPI, top recommendations, revenue chart |
| `/recommendations?runId=` | Per-PPG recommended prices, revenue lift, and actions |
| `/simulate?runId=` | What-if simulator — drag price sliders, units / revenue / guardrails update in real time |
| `/validation?runId=` | Model quality — benchmark comparisons, hold-out WAPE, validation table |
| `/methodology?runId=` | Plain-English explanation of how the pipeline works |

Theme (light/dark) and accent colour are togglable in the top nav.

---

## User flows

### 1. Quick smoke run (no gates, no LLM spend)

1. `make seed` to generate `data/synthetic.csv`.
2. Open `http://localhost:3000`.
3. Select **Server-side path**, leave the default `data/synthetic.csv`.
4. Uncheck **Agent mode** (dry-run) and **Approval gates**.
5. Optionally uncheck **Pick modelling grain** for fully unattended execution.
6. Click **Run pipeline** → all 14 agents complete automatically.
7. When done, click **Business view ↗** in the run subnav to see the presentation-ready summary.

### 2. Interactive run with approval gates

1. Upload your CSV or use the synthetic panel.
2. Enable **Approval gates** and **Pick modelling grain after ingestion**.
3. Click **Run pipeline**.
4. **Gate 1 — Grain selection (after `ppg_mapping`):** the pipeline pauses. Choose a primary modelling grain from a 3×2 grid (PPG / category / brand × chain / store) showing expected cell counts and a recommendation badge. Optionally queue comparison grains for a side-by-side fan-out. Click **Approve**.
5. **Gate 2 — Modeling review (after `modeling`):** inspect per-PPG fit diagnostics. Click **Approve** to continue or **Reject** to mark the run failed.
6. **Gate 3 — Optimization review (after `optimization`):** review the recommendation table and KPI summary. Edit constraints if needed (see flow 4), then click **Approve** to continue to validation and insights.
7. When `insights` completes the executive banner appears with an HTML/PDF download. Open `http://localhost:3001/dashboard?runId=<id>` for the business view.

### 3. Business review of a completed run

1. Note the run ID from the operator timeline (e.g. `abc123`).
2. Open `http://localhost:3001/dashboard?runId=abc123`.
3. Navigate **Dashboard → Recommendations → Simulate → Validation → How it works**.
4. Use **Simulate** to test alternative price points interactively before making a final call.

### 4. Re-running optimization with revised constraints

At the optimization gate, edit any constraint in the approval panel:

| Constraint | Effect |
|---|---|
| **Price ladder** | Comma-separated allowed price points |
| **Margin floor %** | Minimum acceptable gross margin |
| **Competitive gap %** | Required price gap vs. competitor |
| **Max decrease / increase** | Per-PPG price change bounds |
| **Objective** | Optimise for revenue or margin |

Click **Re-run optimization** — the orchestrator re-executes only the `optimization` agent and re-arms the gate. Repeat as needed without restarting ingestion, PPG mapping, or modeling. Click **Approve** when satisfied.

### 5. Comparing modelling grains side-by-side

At Gate 1 (grain selection), after choosing a primary grain tick additional **comparison grains**. The orchestrator fans out a modeling run for each in parallel and merges results into the validation table, letting you compare elasticity recovery and fit metrics across grains before committing.

### 6. Event replay

On any run's timeline page, use the **Replay bar** to scrub to any point in time. All 14 agent cards recompute their state from the visible event slice — useful for debugging or walking through a run step-by-step in a review meeting. `Space` plays/pauses; `←`/`→` steps one event at a time.

## Configuration

The pipeline runs without any environment variables in dry-run mode (LLM
calls fall back to deterministic stubs that produce the same JSON shape).
All runtime config is centralised in `core/config.py` (pydantic-settings)
and accessed via the cached `get_settings()` singleton — don't add ad-hoc
`os.environ.get` calls; add a field to `Settings` instead. `.env` at the
repo root is auto-loaded.

**LLM provider + tracing**

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(unset)_ | When unset, every agent uses its dry-run fallback. Set to enable real Claude API calls. |
| `ANTHROPIC_AUTH_TOKEN` | _(unset)_ | OAuth token from `claude setup-token`; alternative to API key. |
| `LLM_PROVIDER` | _(auto)_ | Force a provider: `dry_run` / `api` / `oauth` / `cli`. Auto-detected from credentials by default; `cli` is never picked implicitly. |
| `LLM_DRY_RUN` | `false` | Force every LLM call into the deterministic fallback even when credentials are set. Useful for cheap CI runs. |
| `LLM_TRACE` | `true` | Per-agent `<agent>_llm_trace.json` audit artefact. Set `false` to suppress (e.g. when prompts may carry sensitive row samples and the run dir will be shared). |
| `LLM_CLI_TIMEOUT_SECONDS` | `180` | Timeout for the `cli` provider shelling out to the local `claude` binary. |

**Model routing**

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_MODEL_OPUS` | `claude-opus-4-7` | Model used for Opus-tier agents (see `OPUS_AGENTS` in `core/llm/routing.py`). |
| `ANTHROPIC_MODEL_SONNET` | `claude-sonnet-4-6` | Model used for all other (Sonnet-tier) agents. |
| `MODEL_<AGENT>` | _(unset)_ | Per-agent override, wins over the role default. `<AGENT>` matches the agent's `name` uppercased (e.g. `MODEL_PPG_MAPPING=claude-sonnet-4-6`, `MODEL_OPTIMIZATION=claude-opus-4-7`). |

**API server**

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins. Methods scoped to GET/POST/OPTIONS, headers to `Content-Type` + `Authorization`. |
| `API_AUTH_TOKEN` | _(unset)_ | When set, every route except `/health` requires `Authorization: Bearer <token>`. Unset = open (dev default). Server-side only — never prefix with `NEXT_PUBLIC_` or it lands in the browser bundle. |
| `MAX_UPLOAD_MB` | `200` | Hard cap on `/uploads` payload size. The endpoint also restricts to `.csv` and sanitizes filenames. |
| `RUN_DIR` | `./runs` | Where per-run artefacts (state, events, DuckDB warehouse, JSONs) are written and served from. |
| `BASELINE_DIR` | `core/data/baselines` | Where the GE drift suite reads its baseline snapshot from (write with `automl baseline-create`). |
| `DRIFT_SLACK_PCT` | `0.4` | ±40% slack on numeric-column drift checks. |
| `VALIDATION__SIGN_PASS`, `VALIDATION__WAPE_PASS`, …| see `core/config.py` | Per-PPG validation cutoffs. Nested env keys: `VALIDATION__<FIELD>`. |

**Published elasticity benchmarks**

The validation agent scores every PPG's recovered elasticity against a
published category range and reports it as `benchmark_status ∈ {in_band,
out_band_low, out_band_high, no_benchmark}` on every row of
`validation_table.json`. The summary surfaces on the validation card's
"Benchmark alignment" gate and on `result.outputs` as
`n_in_benchmark` / `n_out_benchmark` / `benchmark_pass_rate`.

The benchmark table is static and bakes in two sources:

- **Hoch, Kim, Montgomery & Rossi (1995),** *Determinants of Store-Level
  Price Elasticity* (JMR 32:1) — the primary reference for any of the
  18 Dominick's categories the paper covered (beer, yogurt, soft drinks,
  cookies, frozen entrees, refrigerated juices, etc.).
- **Bijmolt, van Heerde & Pieters (2005),** *New Empirical Generalizations
  on the Determinants of Price Elasticity* (JMR 42:2) — meta-analysis
  grand mean **−2.62** across 1,851 estimates; provides per-category
  fallbacks for Hoch-uncovered categories.

The numbers, alias map, and provenance string live at
`core/benchmarks/data/elasticity.json`. The lookup logic is at
`core/benchmarks/elasticity.py` (`lookup_category` does an exact / alias
/ substring match; `classify` returns the band status). Each PPG's
category label is sourced from `ppg_mapping_table.json`, so the join is
automatic for any panel — synthetic or Dominick's — that flows through
the standard PPG mapping agent.

To add a new category, append a row to `elasticity.json`; no code change
is needed. The Confidence Forest chart in the client UI also overlays a
global dashed reference line at the Bijmolt grand mean and outlines any
bar whose elasticity sits outside the published band in warning colour.

**Next.js frontend**

| Variable | Default | Purpose |
|---|---|---|
| `API_PROXY_TARGET` | `http://localhost:8000` | Server-side: where `web/app/api/[...path]/route.ts` forwards browser requests. Set at next-server **runtime** (not build time). |
| `API_AUTH_TOKEN` | _(unset)_ | Same token the API enforces; the Next.js proxy injects it as `Authorization: Bearer …` on every forwarded request. Server-only; never `NEXT_PUBLIC_*`. |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Legacy SSR fallback; only the SSR caller in `app/runs/page.tsx` reads it. Browser-side fetches go through same-origin `/api/*`. |

## Tests

`pytest` is not in the project's runtime deps to keep the install lean;
pull it in on the fly with `uv`:

```bash
uv run --with pytest pytest tests/ -q
# 167 collected
```

Highlights of the suite (`tests/unit/`):

- **Config + settings** (`test_config.py`) — defaults match the pre-refactor
  literals; `MODEL_<AGENT>` overrides flip per-agent routing;
  `ALLOWED_ORIGINS` parses CSV; `BASELINE_DIR` propagates to ingestion.
- **PPG clustering** (`test_ppg_clustering.py`) — synthetic panel through
  dbt + the clusterer; asserts ≥ 95 % SKU agreement vs `synthetic/truth.json`
  (currently 100 %, 48 / 48).
- **Modeling stack** (`test_modeling.py`, `test_lightgbm_model.py`,
  `test_shap_attribution.py`, `test_bayes_hier.py`, `test_predictor.py`) —
  log-log + semi-log sign-retry + LightGBM selection, SHAP per-row identity,
  empirical-Bayes shrinkage bracketed by point & μ̂, shared `Predictor`
  abstraction.
- **Decomposition + simulation** (`test_decomposition.py`,
  `test_ablation_decomp.py`, `test_simulation.py`,
  `test_lightgbm_grid_and_milp.py`) — closed-form + ablation paths
  reconcile to predicted within 1e-6; simulator unit curves monotone in
  price.
- **Optimization + validation** (`test_optimization.py`,
  `test_validation.py`, `test_gate_rerun.py`) — MILP feasibility, soft-relax,
  `chosen_slacks`, rolling-origin CV verdicts, rerun gate loop.
- **Insights + report** (`test_insights.py`) — HTML/PDF render, cost rollup,
  dry-run fallback.
- **Feature refine** (`test_feature_refine.py`) — VIF < 10, |corr| ≤ 0.95,
  `log_price` retained.
- **Chart-data builders** (`test_charts.py`, `test_graceful_degradation.py`)
  — every builder against a synthetic warehouse + missing-column placeholder
  artefacts.
- **LLM trace** (`test_llm_trace.py`) — captures system / user / response;
  `LLM_TRACE=false` cleanly suppresses; failures-after-LLM still flush.
- **API route + artifacts** (`test_artifacts_route.py`, …) — path-traversal
  block uses `pathlib.is_relative_to` so it works on Linux + Windows.
- **Gitignore guard** (`test_gitignore_sources.py`) — fails fast if any
  source path under `core/`, `api/`, `cli/`, `synthetic/`, `tests/`,
  `web/app/`, `web/components/`, `web/lib/` is silently matched by a
  `.gitignore` rule (catches the unanchored `runs/` trap).
- **Dominick's loader** (`test_dominicks_loader.py`) — fixture-based
  round trip from per-category movement + UPC CSVs into the canonical
  panel schema; verifies the 1989-09-14 week anchor, per-unit price
  derivation, promo / `base_price` relationship, and `PanelRow`
  pydantic validation.
- **Elasticity benchmarks** (`test_elasticity_benchmarks.py`,
  `test_validation_benchmark_check.py`) — table loads with known Hoch
  + Bijmolt categories; lookup resolves exact keys, aliases, and
  substring matches; `classify` returns the right band status for
  in-band / more-elastic / less-elastic / no-benchmark / NaN cases;
  validation agent's PPG → category join reads
  `ppg_mapping_table.json` correctly.

## Repository layout

```
api/                   FastAPI app + routes (runs, events, uploads, artefacts, approvals, rerun)
cli/                   Typer CLI (`automl run|seed|baseline-create|prepare-dominicks`)
core/
  agents/              One file per agent; all inherit core.agents.base.Agent
  benchmarks/          Static published-elasticity table (Hoch 1995 + Bijmolt 2005)
                       and the lookup / classify helpers the validation agent uses
  config.py            Central pydantic-settings Settings + get_settings() singleton
  data/                Ingestion, dbt runner, GE runner, profiling tools,
                       chart-ready data builders (charts.py), ingestion report
    loaders/           Third-party dataset adapters; `dominicks.py` lands the
                       Kilts Dominick's archive in the canonical panel schema
  decomp/              Closed-form (due_to.py) + ablation decomposition + group mapping
  features/            EDA tools, engineering pipeline, VIF + correlation refine
  llm/                 AnthropicClient + per-agent model routing + LLM trace + cost
  models/              ElasticityFit base, log-log / semi-log / LightGBM, SHAP,
                       empirical-Bayes shrinkage, shared Predictor abstraction
  optimization/        Constraints, predict helpers, scipy continuous, PuLP MILP
  orchestrator/        RunState, EventBus, gates (incl. rerun loop), async runner
  ppg/                 Per-SKU features, clustering, scoring
  report/              Jinja HTML template + WeasyPrint PDF renderer
  simulation/          Price × promo grid sweeps (OLS + Predictor)
  validation/          Rolling-origin CV + pass/warn/fail check rules
dbt/automl_dbt/        dbt project (DuckDB profile, staging + panel mart, tests)
docs/                  Long-form references (stage-faqs.md, architecture.png)
synthetic/             Synthetic data generator + ground-truth JSON
tests/unit/            pytest suites (167 collected)
web/
  app/                 Next.js routes (`/`, `/runs`, `/runs/[id]`, `/dev/*` mockups)
                       Server-side proxy at `app/api/[...path]/route.ts`
  components/
    charts/            ECharts wrappers (CoverageHeatmap, TrendChart, PPGScatter,
                       PPGPriceBox, EligibilityBars, CorrHeatmap, VIFBar,
                       FeatureHistograms, SHAPBar, ElasticityForest,
                       FittedVsActual, DecompStackedArea, SimulationHeatmap,
                       ConstraintBinding, ResidualHistogram)
    tables/            DataPreview, SchemaTable, QualityPanel, AnomalyTable,
                       DropLog, TargetRelationship, CandidatesTable,
                       RecommendationTable, ValidationTable, InsightsSummary,
                       ResultsTable (shared sortable/expandable base)
    AgentCard.tsx      Per-agent disclosure tile (mounts AgentFAQ)
    AgentFAQ.tsx       Common-questions + corner-cases disclosure (per stage)
    AgentVisuals.tsx   Per-agent inline chart container
    AgentThinking.tsx  Collapsible LLM-trace panel (system / user / response)
    ConstraintEditor.tsx  Inline editor that drives the rerun gate loop
    CostDashboard.tsx  Per-agent token/cost rollup
    ExecutiveBanner.tsx Top-of-run KPIs + HTML/PDF download
    ReplayBar.tsx      Run-replay scrubber with keyboard shortcuts
    RunSidebar.tsx     Cross-run sidebar
    PPGTabs.tsx        Inline tab strip used inside the PPG mapping card
    PPGTable.tsx       PPG → SKU breakdown
  lib/
    agent-faqs.ts      Per-stage FAQ data (single source of truth for AgentFAQ)
    agent-meta.ts      Per-agent metadata + capability sets + summarisers
    api-config.ts      getApiBase() + getAuthHeaders() (browser vs SSR)
    api.ts             Fetch wrappers built on api-config
    chart-config.ts    Shared ECharts defaults (height, grid, fonts, colours)
    theme.ts           STATUS_STYLE / STATUS_DOT / PHASE_COLOR token map
    types.ts           AGENT_ORDER + RunEvent / AgentState / PPG types
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
- CORS is scoped to `ALLOWED_ORIGINS` (default `http://localhost:3000`) with
  methods limited to GET/POST/OPTIONS — set the env to your front-end's
  public origin in production.
- Set `API_AUTH_TOKEN` to require `Authorization: Bearer <token>` on every
  route except `/health`. Same token goes to the Next.js process so its
  proxy injects it automatically.

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
RUN corepack enable && pnpm build

FROM node:20-alpine
WORKDIR /app
COPY --from=build /app ./
EXPOSE 3000
# API_PROXY_TARGET + API_AUTH_TOKEN are read at next-server startup, not
# bake time — set them via env at deploy / docker run.
CMD ["node_modules/.bin/next", "start", "-p", "3000"]
```

`API_PROXY_TARGET` and `API_AUTH_TOKEN` are read at **runtime** by the
Next.js server (no rebuild needed to point at a new backend or rotate the
token). The client bundle never carries either value — browser fetches go
through same-origin `/api/*` and the proxy injects the bearer token
server-side.

### docker-compose sketch

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
      ALLOWED_ORIGINS: http://localhost:3000
      API_AUTH_TOKEN: ${API_AUTH_TOKEN:-}
    volumes:
      - ./runs:/app/runs
      - ./data:/app/data
  web:
    build:
      context: .
      dockerfile: web/Dockerfile
    environment:
      API_PROXY_TARGET: http://api:8000
      API_AUTH_TOKEN: ${API_AUTH_TOKEN:-}
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

## Caveats

### LightGBM extrapolation past the training-price envelope

LightGBM is a tree ensemble. Past the price range it actually saw during
training, every tree flat-lines at its boundary leaf. That has two
follow-on effects the system mitigates but cannot fully eliminate:

1. **Wrong-sign elasticities on noisy data.** Without help, the booster
   is free to learn locally-positive slopes in dense regions of the
   training set, producing a positive own-price elasticity that the
   downstream optimiser then chases. **Mitigation:** `fit_lightgbm`
   (and the `Predictor` refit it shares with simulation / optimisation
   / validation) sets `monotone_constraints=[-1, 0, ...]` so the model
   is mathematically prevented from learning an increasing relationship
   between `log_price` and `log_units`. See
   `core/models/lightgbm_model.py`.

2. **Implausible recommendations past the observed price range.** Even
   with the monotone constraint, predictions stay flat outside the
   training envelope — so the MILP can pick a ladder rung at the top
   end of the configured ladder and claim "no units lost". **Mitigation:**
   the optimisation agent clips the per-PPG price ladder to the
   training-price envelope (`[min, max]` of observed `price` for that
   PPG) for LightGBM winners only. OLS winners extrapolate cleanly so
   they keep the full ladder. Clipped runs surface in the UI via an
   "envelope" chip on the recommendation row and a top-level
   `n_envelope_clipped` count on the optimisation card. See
   `core/agents/optimization.py:_clip_ladder_to_envelope`.

Neither fix is required for OLS winners (closed-form, sign-stable,
well-defined extrapolation). Both apply only to LightGBM. If you raise
`max_decrease` / `max_increase` aggressively for a LightGBM PPG you may
see more rungs dropped — that is the system telling you it doesn't
trust the model that far past the data.
