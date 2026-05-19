# Build Progress

The build ships in 7 incremental phases. Each phase produces a runnable end-to-end slice so progress is demoable at every step.

## 🔔 Open follow-ups (cross-phase, USER-REQUESTED REMINDERS)

These are not blocking the phase plan — they're product-quality reminders
to revisit once the modelling spine is in place. Treat them as a backlog
the UI / modelling phases pull from.

- **Build a better UI.** The current `web/` is functional but spartan.
  Once Phase 6 starts (Insights + Polish), the run page needs: an
  executive-summary card at the top, a polished dark mode, agent-card
  status pills with better empty / error / approval-required states,
  navigation between runs without a full reload, and consistent
  typography / spacing across all agent cards.
- **Improve the modelling solution.** Phase 3a–3c land the iterative
  log-log → semi-log → LightGBM → PyMC stack, but the headline metrics
  (sign recovery, magnitude band) are loose. Tighten: per-PPG
  cross-validation (rolling-origin), elasticity confidence intervals
  surfaced in the table, automatic feature interactions for the LightGBM
  fitter, and a Bayesian-shrinkage option for small-N PPGs.
- **More graphs in earlier and later phases.** Phase 2a-1/2a-2 added
  charts for ingestion / PPG / EDA / features. The phases on either side
  are still chart-poor:
  - **Earlier:** ingestion should add a SKU-count-by-region map and a
    promo-flag-by-week stacked bar; feature_selection should add a
    coverage-vs-target heat-strip per candidate.
  - **Later:** modelling should add a per-PPG fitted-vs-actual scatter
    and a coefficient forest plot; decomposition should add a stacked
    area for due-to over time; simulation should add a 2-D price/promo
    contour plot per PPG; optimisation should add a constraint-binding
    bar; validation should add a hold-out residual histogram.
- **Output results should be shown in a table.** Every agent card
  currently leads with prose + charts. Each card should ALSO render a
  compact, sortable HTML table summarising the agent's per-PPG (or
  per-feature, or per-check) output. The results_reasoning agent
  already writes `model_choice_summary.json` (one row per PPG); make
  that the template — every agent should produce a `*_summary.json`
  matching the same row-shape contract so the same `<ResultsTable>`
  component can render it.

When tackling these, prefer extracting a shared `ResultsTable.tsx` (with
sortable headers, sticky first column, severity colour-coding) over
bespoke per-agent tables — that's the highest-leverage change for the
"results in a table" reminder.

---


## Phase 0 — Scaffolding ✅
Discard `app.py`/`requirements.txt`. Add `pyproject.toml`, repo folders, Pydantic `RunState`/`AgentResult`/`ArtifactRef`, Pandera schema, synthetic data generator + seed script, `AnthropicClient` (with prompt caching + dry_run), orchestrator skeleton with SSE, FastAPI shell, Next.js shell rendering mocked events.

**Status:** complete. Merged in PR #1.

## Phase 1 — Ingestion / PPG Mapping / PPG Selection ✅
Data tools + PPG tools, three agents, approval gate after mapping, PPGTable UI with confidence + rationale. Verify ≥95% SKU agreement vs synthetic truth.

**Status:** complete. PPG mapping recovers 100% of the synthetic truth (8/8 PPGs, 48/48 SKUs).

## Phase 2 — EDA + Feature Engineering + Refine ✅
EDA tools + feature tools, four agents (feature-candidates, EDA, feature-engineering, feature-refine), artifact gallery UI. Verify VIF<10 and no |corr|>0.95 pairs remain.

**Status:** complete. Refined feature set keeps `log_price` as the elasticity primary; max VIF 7.92, max |corr| 0.91 on the synthetic panel.

## Phase 2a — Data Visibility Layer
Make every Phase-1/2 agent's work visible. A reviewer must be able to *see*
what the data looks like, which quality rules fired and why, how SKUs got
grouped, and which feature decisions got made — not just whether each agent
returned `done`.

**Layout:** inline — each AgentCard expands with its own visuals (mockups
under `web/app/dev/`).
**Charts:** Apache ECharts via `echarts-for-react`, tree-shaken
(core + heatmap + scatter + boxplot + bar + line modules only),
each chart wrapped in `"use client"` + `dynamic({ ssr: false })`.
**Architecture:** per-agent — each agent writes its own chart-ready artefacts
at the end of `_execute()`. Shared math lives in `core/data/charts.py`;
agents call it. No new agent in the DAG.
**Shipping:** split into **2a-1** (quality + PPG rationale) and **2a-2**
(EDA + features + LLM trace) so each is one day, one merge.

### Cross-cutting decisions
- **dbt + GE results parser** (`core/data/test_results.py`): single normalised
  list of `{source, rule, severity, status, message, row_count}` joined from
  dbt's `target/run_results.json` and GE's validation output. Both 2a-1 and
  the existing ingestion artefact consume it.
- **LLM trace:** full system + user + response per agent into
  `<agent>_llm_trace.json`. Default on; disable per-run with
  `LLM_TRACE=false`. Trace is also written when dry-run fired, with a
  `dry_run: true` field so the UI can show the deterministic fallback message.
- **Real-data graceful degradation:** PPG visuals depend on `brand` /
  `category` / `pack_size`. When any are absent in the uploaded CSV, the
  per-chart component renders a `"missing column: <name>"` placeholder
  rather than crashing. Acceptance criterion in both sub-phases.
- **Test stack:** no new project dep. The throwaway
  `scripts/visual_smoke.mjs` Playwright script (already used for manual
  screenshots) gets promoted into the repo and called from
  `make test-visual`. CI stays pytest-only; visual smoke is opt-in until a
  later phase formalises e2e.
- **PPG visual:** ship all three (`tier × log-price`, behaviour-based,
  faceted brand × pack-size by category) inside one inline tab strip within
  the PPG card. The tabbed-inline pattern (`PPGTabs.tsx`) is a generic
  component we'll reuse for any future "multiple views of the same thing"
  surface.

---

### Phase 2a-1 — Quality story + PPG rationale ✅
**Status:** complete. End-to-end run on the synthetic panel produces all
eight chart-ready artefacts; the run page renders coverage heatmap +
weekly trend + quality panel + anomaly list under ingestion, three
tabbed scatter views + price-box + full PPG breakdown under PPG mapping,
and stacked eligibility bars under PPG selection. `/runs/[id]` first-load
JS ≈ 90 kB gz with ECharts loaded (target was < 250 kB). Full pytest sweep:
16 passed, including the new graceful-degradation case.

**Backend**
- `core/data/charts.py` — builders:
  - `coverage_grid(con)` → sparse SKU × week presence matrix
  - `weekly_trend(con)` → panel-wide units + price + promo share by week
  - `ppg_scatter_tier(assignments, con)` → x = tier, y = log price
  - `ppg_scatter_behaviour(assignments, con)` → x = log mean units, y = per-SKU corr(log units, log price)
  - `ppg_scatter_facet(assignments, con)` → faceted (category, brand × pack-size) coordinates
  - `ppg_price_box(con)` → per-PPG quantiles
  - `eligibility_bars(selection)` → stacked-bar dataset
- `core/data/test_results.py` — dbt + GE results parser.
- Extend `core/agents/ingestion.py`: write `coverage_grid.json`,
  `weekly_trend.json`, `quality_results.json`; enrich `ingestion_findings`
  with severity + row counts.
- Extend `core/agents/ppg_mapping.py`: write three `ppg_scatter_*.json`
  + `ppg_price_box.json`.
- Extend `core/agents/ppg_selection.py`: write `ppg_eligibility_bars.json`.

**Frontend**
- Install `echarts` + `echarts-for-react`; tree-shake; remove `recharts`
  from real components (mockups under `web/app/dev/` keep it).
- `web/components/charts/EChart.tsx` — typed `option` wrapper, dark slate
  theme, `ssr: false`.
- Chart components: `TrendChart`, `CoverageHeatmap`, `PPGScatter`
  (parametric — takes any of the three datasets), `PPGPriceBox`,
  `EligibilityBars`.
- Tables: `DataPreview`, `SchemaTable`, `QualityPanel` (pass/warn/fail
  pills + severity + row-count), `AnomalyTable`.
- `web/components/PPGTabs.tsx` — inline tab strip for the three PPG views.
- Inline integration in `AgentCard.tsx`:
  - **ingestion** → DataPreview + SchemaTable + CoverageHeatmap +
    QualityPanel + AnomalyTable.
  - **ppg_mapping** → PPGTabs (3 scatters) + PPGPriceBox; move the existing
    standalone PPGTable into this card body.
  - **ppg_selection** → EligibilityBars.

**Tests**
- `tests/unit/test_charts.py` — every builder returns non-empty,
  well-formed shapes on the synthetic warehouse.
- `tests/unit/test_quality_results.py` — parser merges dbt + GE outputs into
  the normalised list.
- `tests/unit/test_graceful_degradation.py` — feed the chart builders a
  panel missing `brand` / `category` / `pack_size`; verify a structured
  `{"missing_columns": [...]}` artefact is produced instead of an exception.
- `scripts/visual_smoke.mjs` (committed) — boots api + web, drives a no-gate
  run, asserts ingestion + ppg_mapping cards expand and chart containers
  have non-zero dimensions.

**Acceptance gate**
- Ingestion card surfaces quality results with severity, message, row count.
- All 3 PPG scatters render with non-empty data; price box shows 8 PPGs.
- Tab switching between the 3 PPG views works (no remount jank).
- Missing-column case shows a placeholder, not a crash.
- `/runs/[id]` first-load JS < 250 kB gz.
- pytest + visual smoke green.

---

### Phase 2a-2 — EDA, features, LLM trace ✅
**Status:** complete. EDA card shows the weekly trend + correlation
heatmap + target-relationship table + findings. Feature-engineering card
shows a 16-tile histogram grid (one per engineered column, with μ / σ /
n). Feature-refine card shows the VIF bar (threshold marker, red /
amber / green per VIF bucket), the refined-set correlation heatmap,
the drop log + kept-list pills. All seven LLM-using agents emit an
`<agent>_llm_trace.json` capturing system / user / response / model /
tokens / `dry_run`; `LLM_TRACE=false` cleanly disables capture (covered
by unit test). The "Agent thinking" panel renders each call collapsible
with a `dry-run` badge or live token cost. `/runs/[id]` first-load JS:
317 kB raw (≈ 90 kB gz; target was < 350 kB gz). Full pytest sweep:
22 passed.

**Backend**
- `core/data/charts.py` — additions: `feature_histograms`, `corr_refined`.
- Extend `core/agents/eda.py`: write chart-ready slices of
  `eda_report.json` so the frontend doesn't reshape arrays.
- Extend `core/agents/feature_engineering.py`: write
  `feature_histograms.json` (20-bin histogram + mean/std per engineered
  column).
- Extend `core/agents/feature_refine.py`: write `corr_refined.json`
  (refined-set correlation matrix).
- `core/agents/base.py`: LLM-trace capture in `call_llm()`. Records system,
  user, response, model, cache hit/miss, `dry_run` flag. Writes
  `<agent>_llm_trace.json` unless `LLM_TRACE=false`.

**Frontend**
- Chart components: `CorrHeatmap`, `VIFBar`, `FeatureHistograms`.
- Tables: `DropLog` (feature + reason).
- `AgentThinking.tsx` — collapsible 3-pane (system / user / response) with a
  `dry-run` badge.
- Inline integration:
  - **eda** → TrendChart + target-relationship table + CorrHeatmap.
  - **feature_engineering** → FeatureHistograms grid.
  - **feature_refine** → VIFBar + CorrHeatmap (refined) + DropLog.
  - **every LLM-using agent** → AgentThinking sub-section under its
    reasoning/tool-call detail.

**Tests**
- `tests/unit/test_llm_trace.py` — base Agent writes the trace when
  `LLM_TRACE` is unset / `"true"`, skips cleanly when `"false"`; dry-run
  fallback writes a trace with `dry_run: true`.
- Visual smoke extended to expand eda + feature_engineering + feature_refine
  cards.

**Acceptance gate**
- Every Phase 1/2 agent shows at least one chart in its expanded card.
- LLM trace section renders in both live and dry-run modes with the right
  badge; `LLM_TRACE=false` produces no trace files.
- Bundle < 350 kB gz for `/runs/[id]` with ECharts loaded.
- Existing 7 + 4 new unit tests pass; visual smoke green.

---

**Status:** planned (2a-1 and 2a-2 ready to start). Mockups in
`web/app/dev/` are the visual reference until real components land.

## Phase 3 — Modeling + Results Reasoning
Model tools (log-log, semi-log, LightGBM+SHAP, PyMC hierarchical), iterative modeling agent that retries on wrong-sign elasticities, results-reasoning agent, model-choice approval gate, elasticity chart UI. Verify log-log recovers truth on synthetic.

**Status:** in progress.

### Phase 3a — Log-log + semi-log with sign-retry ✅
**Status:** complete. Modeling agent now ships for real (replaces
StubAgent). Log-log OLS recovers the correct elasticity sign for 8/8 PPGs
on the synthetic panel; magnitudes land in the plausible [0.3, 6.0] band
on 8/8. The sign-retry to semi-log is wired and verified by a forced
failure (monkeypatched log-log) that the agent recovers from end-to-end.

**Backend**
- `core/models/base.py` — `ElasticityFit` dataclass (own elasticity,
  std err, p-value, R², n, controls, coefficients, diagnostics, `sign_ok`).
- `core/models/loglog_ols.py` — statsmodels OLS on
  `log_units ~ log_price + controls`.
- `core/models/semilog_ols.py` — statsmodels OLS on
  `log_units ~ price + controls`; converts β to elasticity at mean price.
- `core/agents/modeling.py` — per-PPG fit + sign-retry loop, writes
  `modeling_results.json` (every attempt) + `elasticity_per_ppg.json`
  (compact). LLM emits narrative + concern flags; dry-run fallback intact.
- `core/orchestrator/runner.py` — registers `ModelingAgent` in
  `REAL_AGENTS`. The `modeling` approval gate is still on by default.

**Tests**
- `tests/unit/test_modeling.py` — log-log sign recovery ≥7/8,
  magnitude-band sanity ≥5/8, semi-log smoke check, retry-wiring test,
  full agent run writes both artefacts.

**Deps**
- Add `statsmodels>=0.14` to `pyproject.toml`.

**Acceptance gate**
- Log-log recovers the elasticity sign on ≥7/8 PPGs (got 8/8). ✅
- Sign-retry fires and switches the winner to semi-log when log-log fails. ✅
- `modeling_results.json` + `elasticity_per_ppg.json` are on disk after
  the agent runs end-to-end. ✅

### Phase 3b — LightGBM + WAPE comparison + results-reasoning agent ✅
**Status:** complete. The modeling agent now fits three candidates per
eligible PPG (log-log, semi-log on sign-retry, LightGBM) on a chronological
80/20 split, ranks them by hold-out WAPE, and writes per-PPG attempts +
winner to `modeling_results.json`. A new `ResultsReasoningAgent` runs
deterministic verdict checks (sign, magnitude band, R² floor, hold-out
WAPE) on the modeling output and writes a flat one-row-per-PPG
`model_choice_summary.json` for the UI table.

**Backend**
- `core/models/metrics.py` — `wape_units` (raw-units WAPE from log-scale
  predictions) + `chronological_split` (time-aware 80/20).
- `core/models/lightgbm_model.py` — LightGBM regressor; elasticity
  recovered via numerical bump on `log_price` (Δ = log(1.01)) averaged
  across rows; feature importances captured for the UI.
- `core/models/loglog_ols.py` + `core/models/semilog_ols.py` — extended
  to accept an optional `test` frame and record `test_wape` in
  `diagnostics`.
- `core/agents/modeling.py` — refactor: per-PPG split, fit all three
  candidates (semi-log only on sign-retry), winner = lowest test WAPE
  among sign-correct fits. `winners_by_family` surfaced in
  `result.outputs`.
- `core/agents/results_reasoning.py` — new agent; reads
  `modeling_results.json`, emits `results_reasoning.json` (verdict per
  PPG with check breakdown) + `model_choice_summary.json` (compact
  table-shaped rows).
- `core/orchestrator/runner.py` — registers `ResultsReasoningAgent` in
  `REAL_AGENTS`. `results_reasoning` is now a real agent in the DAG.

**Tests** (13 new, all green; full suite: 39 passed, 3 skipped)
- `tests/unit/test_lightgbm_model.py` — sign recovery on a clean DGP,
  feature-importance shape, WAPE matches a hand computation.
- `tests/unit/test_results_reasoning.py` — pass / warn / fail verdicts
  per check rule, table summary one-row-per-PPG.
- `tests/unit/test_modeling.py` — selection logic tests (mocked
  fitters): sign-retry path picks lowest-WAPE sign-correct candidate;
  log-log-only path skips semi-log; CSV artefact assertions widened to
  include `lightgbm` + `test_wape`.

**Deps**
- Add `lightgbm>=4.3` to `pyproject.toml`.

**Acceptance gate**
- 3-model pool reported in `modeling_results.model_pool`. ✅
- `test_wape` recorded for every non-skipped winner. ✅
- `results_reasoning.json` + `model_choice_summary.json` written end-to-end. ✅
- Selection logic verified via mocked fitters (no flaky data dependency). ✅

### Phase 3b' — SHAP feature attribution + model-choice approval gate UI (pending)
Carved out of the original 3b plan. Adds SHAP value summaries per PPG
(beeswarm-ready dataset) and wires the existing default `modeling`
approval gate to a model-choice modal in the UI showing the candidates
table from `modeling_results.json`.

### Phase 3c — Bayesian hierarchical + elasticity chart UI (pending)
PyMC hierarchical model partial-pooling across PPGs (Numpyro backend for
compile speed), and the inline elasticity-with-error-bars chart in the
modeling AgentCard. Wires the existing model-choice gate to the UI.

## Phase 4 — Decomposition + Simulation ✅ (backend slice)
Decomp + sim tools, two agents, stacked-bar (due-to) + scenario-grid heatmap UI. Verify decomposition reconciles to observed.

**Status:** backend complete; inline UI charts pending (tracked in
"Open follow-ups → More graphs"). Per-row decomposition reconciles to
the model's prediction within 1e-9 by construction; per-PPG aggregate
reconciliation error stays < 1e-6 on the synthetic panel. Simulation
grid produces monotone-in-price unit response and the expected
boundary revenue maximum for elastic demand.

### Phase 4a — Closed-form decomposition + OLS scenario grid ✅
**Status:** complete.

**Backend**
- `core/decomp/due_to.py` — closed-form per-row decomposition for any
  OLS coefficient dict. Splits each observed week into
  ``base + Σ due-to-feature + residual``. Per-feature contributions
  sum to ``(predicted - base)`` exactly (allocated by log-space share).
  Residual = ``observed - predicted``.
- `core/decomp/groups.py` — explicit feature → business-group mapping
  (price / promo / distribution / competitor / seasonality / lags /
  other). Unknown columns fall through to ``"other"`` so they're
  surfaced rather than dropped.
- `core/simulation/grid.py` — vectorised closed-form price × promo
  sweep. 15 price multipliers × 2 promo states = 30 cells per PPG,
  microseconds per PPG. Reports per-cell units / revenue / margin
  plus the revenue-optimal and margin-optimal cell.
- `core/agents/decomposition.py` — refits the winning OLS family on
  the full feature frame (no holdout — every week must be attributed),
  decomposes, and writes:
  - `decomposition_per_ppg_week.json` — weekly grid per PPG with
    `due_by_group` rolled up per business group.
  - `decomposition_summary.json` — totals + per-feature + per-group
    contributions + reconciliation diagnostic per PPG.
  - `decomposition_table.json` — flat ``(ppg_id, group, due_units,
    share_of_lift)`` rows for the UI's shared `<ResultsTable>`.
- `core/agents/simulation.py` — reads the modelling agent's stored
  coefficients (no refit needed), sweeps the grid per PPG, writes:
  - `simulation_grid.json` — full per-cell grid per PPG.
  - `simulation_summary.json` — best-revenue + best-margin cell per PPG.
  - `simulation_table.json` — flat ``(ppg_id, objective, multiplier,
    promo, value, units)`` rows.
- `core/orchestrator/runner.py` — registers both new agents in
  `REAL_AGENTS`; `decomposition` and `simulation` are now real DAG
  stages.

**LightGBM-winning PPGs** are skipped for now with a structured note
(``result.outputs["skipped"]``). Closed-form decomposition isn't
applicable; ablation-based decomposition + a LightGBM simulator land
in the Phase 4b follow-up.

**Tests** (13 new, all green; full suite 52 passed / 3 skipped)
- `tests/unit/test_decomposition.py` — per-row reconciliation < 1e-9,
  residual identity, group aggregation matches per-feature sum,
  zero-lift edge case, summary reconciliation < 1e-9, agent writes
  three artefacts, LightGBM winner skipped cleanly.
- `tests/unit/test_simulation.py` — units monotone-decreasing in
  price, revenue-optimal at grid boundary for elastic demand, TPR
  lifts units, semi-log grid shape, agent writes three artefacts,
  LightGBM winner skipped cleanly.

**Acceptance gate**
- Decomposition reconciles to predicted within 1e-6 per PPG. ✅
- Simulation produces monotone-in-price unit curves. ✅
- All three artefacts on disk per agent end-to-end. ✅

### Phase 4b — Ablation decomposition + LightGBM simulator (pending)
Adds numerical-ablation decomposition so LightGBM-winning PPGs aren't
skipped, plus a LightGBM-backed simulator (persist trained models in
Phase 3b, or refit in the agent). Also covers the inline stacked-bar
+ contour-heatmap UI charts called out in "Open follow-ups → More
graphs (Later)".

## Phase 5 — Optimization + Validation
Opt tools, constraint-elicitation gate, scipy continuous warm start → PuLP MILP with ladder/margin-floor/comp-gap, validation agent (holdout WAPE, elasticity reasonableness, stability), constraint editor + recommendation table UI.

**Status:** pending.

## Phase 6 — Insights + Report + Polish
Insights agent, HTML + PDF report (jinja + weasyprint), cost dashboard, run replay, dark mode, error/retry states.

**Status:** pending.
