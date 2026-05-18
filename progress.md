# Build Progress

The build ships in 7 incremental phases. Each phase produces a runnable end-to-end slice so progress is demoable at every step.

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

### Phase 3b — LightGBM + SHAP + results-reasoning agent (pending)
Adds LightGBM + SHAP per PPG so the agent can compare a non-parametric fit
to the parametric ones, picks the best-by-WAPE model, and surfaces SHAP
attribution. Plus the results-reasoning agent + model-choice approval gate.

### Phase 3c — Bayesian hierarchical + elasticity chart UI (pending)
PyMC hierarchical model partial-pooling across PPGs (Numpyro backend for
compile speed), and the inline elasticity-with-error-bars chart in the
modeling AgentCard. Wires the existing model-choice gate to the UI.

## Phase 4 — Decomposition + Simulation
Decomp + sim tools, two agents, stacked-bar (due-to) + scenario-grid heatmap UI. Verify decomposition reconciles to observed.

**Status:** pending.

## Phase 5 — Optimization + Validation
Opt tools, constraint-elicitation gate, scipy continuous warm start → PuLP MILP with ladder/margin-floor/comp-gap, validation agent (holdout WAPE, elasticity reasonableness, stability), constraint editor + recommendation table UI.

**Status:** pending.

## Phase 6 — Insights + Report + Polish
Insights agent, HTML + PDF report (jinja + weasyprint), cost dashboard, run replay, dark mode, error/retry states.

**Status:** pending.
