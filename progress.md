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

### Phase 3b' — SHAP feature attribution + model-choice approval gate UI ✅
**Status:** complete. Per-PPG SHAP-style attribution now lands on disk for
every winner; the modeling AgentCard renders a sortable candidates table
(winner row marked, every attempt expandable) plus a per-PPG mean-|SHAP|
bar chart. The default `modeling` approval gate already pauses the run;
this slice fills the previously-empty "what am I approving?" space with
the candidates table + SHAP bar before the user clicks Approve/Reject.

**Backend**
- `core/models/shap_attribution.py` — two paths producing the same JSON
  shape (`{base_value, mean_abs_shap, mean_shap, beeswarm, ...}`):
  - LightGBM uses native `predict(X, pred_contrib=True)` — exact
    tree-SHAP with no extra dependency.
  - Log-log + semi-log use the linear identity centred on the train mean
    (`shapᵢ = βᵢ·(xᵢ - x̄ᵢ)`), so `base + Σ shap == ŷ` row-wise.
- Each fitter now stuffs a `diagnostics["shap"]` summary at fit time so
  the agent doesn't have to refit just for attribution.
- `core/agents/modeling.py` — `_collect_shap()` rolls the winner's SHAP
  into a flat `shap_per_ppg.json` artifact; `result.outputs["n_shap"]`
  surfaces the count on the AgentCard.

**Frontend**
- `web/components/charts/SHAPBar.tsx` — mean-|SHAP| horizontal bar; tint
  green / blue by mean signed SHAP (positive / negative pressure on
  log-units).
- `web/components/tables/CandidatesTable.tsx` — per-PPG candidates table.
  Click a row to select the PPG for the SHAP panel; expand to see all
  attempted models (loglog / semilog-on-retry / lightgbm) with elasticity,
  R², train + test WAPE, sign-ok flag. Winner row marked.
- `web/components/AgentVisuals.tsx` — new `ModelingVisuals` panel ties
  the table to the bar chart via local PPG selection state.
- `web/components/AgentCard.tsx` — adds `modeling` to the visuals +
  thinking sets so the panel surfaces alongside the existing
  awaiting-approval footer.
- `web/lib/agent-meta.ts` — modeling card now shows `n/n correct sign`,
  retry count, SHAP-summary count chips.

**Tests** (6 new + 1 extended; full unit suite: 73 passed)
- `tests/unit/test_shap_attribution.py` — per-row reconstruction identity
  for OLS (exact) and LightGBM (within 1e-6), mean |SHAP| sorted desc,
  beeswarm sample cap, dominant-feature ranking on a clean DGP.
- `tests/unit/test_modeling.py` — extended to assert `shap_per_ppg.json`
  is written for every fit PPG with the expected sorted-shap shape.

**Follow-up surfaced + resolved**
- The orchestrator's modeling agent was seeing zero rows per PPG on the
  end-to-end smoke because `ppg_mapping.json` emitted `PPG_AUTO_*` IDs
  while `main.panel.ppg_id` (the source for `ppg_week_aggregate`) was
  still the synthetic-truth labels (`PPG01..08`). Unit tests passed
  because they reseed `ppg_selection.json` with matching IDs.
  **Fixed:** the mapping agent now calls
  `core.ppg.cluster.apply_mapping_to_panel` after clustering, which
  rewrites `main.panel.ppg_id` from the SKU → PPG_AUTO_* assignments.
  End-to-end smoke now produces real fits — modelling recovers correct
  elasticity sign for 8/8 PPGs, results-reasoning passes 6/8,
  hierarchical posterior pools 4 OLS winners with τ²≈1.1, decomposition
  + simulation produce real artefacts for the OLS-winning PPGs.

### Phase 3c — Empirical-Bayes shrinkage + forest plot UI ✅
**Status:** complete. Per-PPG OLS estimates are pooled with closed-form
empirical-Bayes / Stein shrinkage; the modeling AgentCard now renders a
forest plot showing OLS point ± 95% CI alongside the shrunken posterior
± 95% CI, with the population mean μ̂ drawn as a reference line. No
PyMC / no MCMC — DerSimonian-Laird method-of-moments τ² estimator runs
in sub-millisecond time and is fully deterministic.

**Backend**
- `core/models/bayes_hier.py` — `shrink()` implements the random-effects
  meta-analysis model ``β̂ᵢ | βᵢ ~ N(βᵢ, sᵢ²)``, ``βᵢ ~ N(μ, τ²)``. μ̂ is the
  inverse-variance weighted mean; τ̂² is the DerSimonian-Laird MoM
  estimator clamped at 0; each PPG's posterior is the inverse-variance
  combination of likelihood and prior. Returns ``HierarchicalPosterior``
  with point, shrunk_mean, 95% CI, shrinkage_weight per PPG.
- `core/agents/modeling.py` — pools only OLS winners (LightGBM's
  std_err is a row-dispersion, not a sampling SE, so it's excluded);
  writes `hierarchical_posterior.json`; surfaces ``n_shrunk`` and
  ``tau_squared`` in ``result.outputs``.

**Frontend**
- `web/components/charts/ElasticityForest.tsx` — custom ECharts forest
  plot. Two series per PPG (OLS in grey, shrunken posterior in
  emerald), sorted by OLS point estimate. Vertical zero line + dotted
  μ̂ reference line; tooltip reports CI and shrinkage weight.
- `web/components/AgentVisuals.tsx` — forest plot rendered above the
  SHAP panel when posterior is non-empty.
- `web/lib/agent-meta.ts` — modeling card now shows pooled-PPG count
  and τ² alongside sign / retry / SHAP chips.

**Tests** (8 new + 1 extended; full unit suite: 66 passed, 3 skipped)
- `tests/unit/test_bayes_hier.py` — τ²=0 collapses to pooled mean,
  real heterogeneity yields τ²>0 and partial shrinkage, noisy PPGs
  shrink more than precise ones, posterior lies between point and
  μ̂, CI is symmetric and uses ±1.96 z, non-finite/zero-SE rows
  dropped, empty input returns nan population, payload exposes
  required keys.
- `tests/unit/test_modeling.py` — extended to assert
  `hierarchical_posterior.json` is written, contains only OLS-winner
  PPGs, and each shrunk_mean is bracketed by the point estimate and μ̂.

**Acceptance**
- Empirical-Bayes runs deterministically in sub-second time, no new
  heavy deps (uses numpy only). ✅
- Forest plot renders shrinkage overlay against OLS baseline. ✅
- LightGBM winners excluded from the pool (different SE semantics). ✅

## Phase 4 — Decomposition + Simulation ✅
Decomp + sim tools, two agents, stacked-bar (due-to) + scenario-grid heatmap UI. Verify decomposition reconciles to observed.

**Status:** complete (4a + 4b shipped). Inline UI charts still on the
"Open follow-ups → More graphs" backlog. Per-row decomposition reconciles
to the model's prediction within 1e-9 by construction; per-PPG aggregate
reconciliation error stays < 1e-6 on the synthetic panel across **all 8
PPGs** (was 4 before — Phase 4b unblocked the LightGBM winners).

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

### Phase 4b — Ablation decomposition + LightGBM simulator/optimiser ✅
**Status:** complete. LightGBM-winning PPGs are no longer dropped from
the downstream pipeline. End-to-end on the synthetic panel: 4 OLS + 4
LightGBM winners — all 8 now flow through decomposition, simulation,
optimisation, and validation. Recommended revenue lifts from ~$378k
(4 PPGs) to ~$589k (8 PPGs) on the synthetic dataset.

**Backend**
- `core/models/predictor.py` — shared `Predictor` abstraction. OLS path
  evaluates the closed-form `α + Σ βᵢ·xᵢ` from saved coefficients;
  LightGBM path refits the booster on the PPG's train slice
  (deterministic via `random_state=0`) and wraps the trained estimator.
  Both expose `predict_log` / `predict_units` so downstream agents stop
  branching on `winner_model`.
- `core/decomp/ablation.py` — group-wise ablation decomposition for
  non-linear predictors. For each row: `pred_log =
  predictor(observed)`, `base_log = predictor(reference)`, per-group
  `delta_log = predictor(group_on, others_ref) - base_log`; allocates
  unit lift across groups by `delta_log` share. Mirrors the OLS
  reconciliation contract (`base + Σ due_group ≈ predicted`).
- `core/simulation/grid.py` — adds `simulate_predictor_grid()` that
  drives any `Predictor` through the price × promo sweep with the same
  output shape as `simulate_ols_grid`.
- `core/optimization/predict.py` — adds `cell_metrics_via_predictor()`
  and `predict_units_via_predictor()` so the scipy + PuLP solvers can
  score LightGBM cells the same way they score OLS cells.
- `core/optimization/constraints.py` — `PPGOptInputs.predictor` field
  lets the agent pass a fitted predictor without changing the OLS
  call shape (existing OLS callers leave it `None`).
- `core/optimization/continuous.py` + `milp.py` — route every cell
  scoring through `_cell_metrics_for()` which dispatches on
  `inp.predictor`.
- `core/validation/rolling.py` — `fit_one_fold()` now handles
  `model_kind="lightgbm"` (refits the booster per fold).
- `core/agents/{decomposition,simulation,optimization,validation}.py`
  — extend the `SUPPORTED_MODELS` set to include `"lightgbm"`; build
  predictor via `core.models.predictor.build_predictor` for the new
  branch; decomposition tags each summary with
  `attribution_method ∈ {closed_form, ablation}` so the UI can label.

**Tests** (3 new files + flipped 4 existing skip-assertions; full unit
suite 135 passed)
- `tests/unit/test_predictor.py` — OLS predictor matches closed-form
  exactly; LightGBM predictor's bump-elasticity is negative on a clean
  DGP; `test_ratio=0.0` trains on the full frame; OLS path uses saved
  coefficients without refitting.
- `tests/unit/test_ablation_decomp.py` — reference frame zeros dummies
  + means continuous features + log_price baselines to log_base_price;
  aggregate reconciliation `< 1e-6`; residual identity holds; zero
  lift when at reference; price drives negative lift when above base.
- `tests/unit/test_lightgbm_grid_and_milp.py` — LightGBM grid units
  trend monotone-decreasing in price (regressed log-log slope < -0.5);
  single-cell scoring matches grid sweep cell-by-cell; MILP picks a
  ladder cell for the LightGBM predictor and feasibility holds.
- Flipped `test_decomposition` / `test_simulation` /
  `test_optimization` / `test_validation` LightGBM-skip assertions to
  LightGBM-flows-through assertions.

**LightGBM extrapolation mitigations (shipped together with 4b core)**
- `core/models/lightgbm_model.py` + `core/models/predictor.py`:
  `monotone_constraints=[-1, 0, ...]` pins ``log_price`` to be
  monotonically decreasing in ``log_units``. Removes the
  wrong-sign-elasticity failure mode entirely.
- `core/agents/optimization.py`: for LightGBM winners only, the price
  ladder is clipped to ``[min_train_price, max_train_price]`` before
  the MILP runs. Dropped rungs are surfaced via
  `optimization_results.json#envelope_clip` and counted by
  `outputs.n_envelope_clipped`; the UI recommendation row shows an
  "envelope" chip when this fired. OLS winners are left unchanged —
  they extrapolate cleanly. Documented under README "Caveats".
- End-to-end on synthetic: 3/4 LightGBM PPGs hit the envelope clip;
  recommendations slide from the previous top-of-ladder 1.15 down
  to 0.98/1.00 inside their training range. Recommended revenue:
  ~$578k (vs the un-clipped $589k).

**Acceptance gate**
- LightGBM-winning PPGs are no longer skipped by any downstream
  stage. ✅
- Ablation decomposition reconciles to predicted within 1e-6 on the
  synthetic panel (matches OLS path). ✅
- LightGBM simulator + MILP solve cleanly end-to-end. ✅
- All 135 unit tests pass; end-to-end synthetic smoke green. ✅

## Phase 5 — Optimization + Validation ✅
Opt tools, constraint-elicitation gate, scipy continuous warm start → PuLP MILP with ladder/margin-floor/comp-gap, validation agent (holdout WAPE, elasticity reasonableness, stability), constraint editor + recommendation table UI.

**Status:** complete (5a + 5a' + 5b shipped).

### Phase 5a — Constrained price optimisation ✅
**Status:** complete. End-to-end run on the synthetic panel optimises every
OLS-winning PPG against a 9-rung price ladder under margin-floor +
competitive-gap + move-guardrail constraints. The scipy continuous solver
runs first as a warm-start anchor; the PuLP MILP then picks the best
ladder rung × promo state. When no cell is strictly feasible the agent
falls back to a soft-constraint relaxation that picks the least-violating
cell and reports which constraints were violated and by how much. On the
synthetic panel: 3/4 PPGs strictly feasible, 1 hits a relaxation because
the SKU's base price ($1.00) sits too far below the competitor reference
($2.13) to satisfy the 15% comp-gap inside the 20% move guardrail.

**Backend**
- `core/optimization/constraints.py` — `OptimizationConstraints` dataclass
  with price ladder, promo states, COG %, margin floor %, comp gap %,
  move guardrail, objective (revenue / margin). Per-PPG `PPGOptInputs`
  carries coefficients + base price + context + competitor reference.
- `core/optimization/predict.py` — single-cell closed-form unit / revenue
  / margin prediction; mirrors `simulate_ols_grid` for parity, used by
  both solvers.
- `core/optimization/continuous.py` — scipy `minimize_scalar` over the
  bounded multiplier interval, intersected with the margin-floor +
  comp-gap windows. Returns the unconstrained-but-bounded optimum per
  promo state and picks the better.
- `core/optimization/milp.py` — PuLP CBC MILP: pre-compute every
  (multiplier, promo) cell's value + constraint slacks, pick exactly one
  feasible cell that maximises the objective. Infeasible problems fall
  back to a relaxed solve that minimises `Σ violation` with the
  objective as a secondary tie-breaker; reports `binding_violations`
  with per-constraint magnitudes.
- `core/agents/optimization.py` — orchestrates per-PPG continuous +
  MILP, honours `run.options["optimization"]` overrides for every
  constraint field. Writes:
  - `optimization_results.json` — full continuous + MILP solution per PPG.
  - `optimization_table.json` — flat `(ppg_id, multiplier, price, promo,
    units, revenue, margin, ...)` rows for the UI's shared
    `<ResultsTable>`.
  - `optimization_constraints.json` — the resolved constraint set
    (defaults + overrides) for audit + UI display.
- `core/orchestrator/runner.py` — registers `OptimizationAgent` in
  `REAL_AGENTS`; replaces the StubAgent. The `optimization` gate stays
  default-on (existing post-run review pattern).
- `core/llm/routing.py` — adds `optimization` to `OPUS_AGENTS` for the
  rationale narration.

**Frontend**
- `web/lib/agent-meta.ts` — optimization card surfaces
  `n_optimised`, objective, ladder size, and `n_relaxed` chips.
- Inline constraint editor + recommendation table land in 5b alongside
  the validation card.

**Tests** (10 new; full suite 94 passed)
- `tests/unit/test_optimization.py` — predict matches simulator grid
  cell-for-cell; continuous picks the lower bound for elastic demand;
  continuous flags infeasible bounds; MILP respects ladder + margin
  floor; MILP relaxes with comp-gap violation when the competitor sits
  outside the move guardrail; MILP picks the upper ladder rung for
  inelastic demand on the margin objective; agent writes three
  artefacts; LightGBM winners skipped; `run.options["optimization"]`
  override resolves into the saved constraint set + recommended cell.

**Deps**
- Add `pulp>=2.8` to `pyproject.toml`.

**Acceptance gate**
- MILP recommendation lies on the configured ladder. ✅
- Margin-floor + competitive-gap + move-guardrail constraints enforced
  in the strict path. ✅
- Soft-constraint fallback fires + surfaces `binding_violations` when
  no cell is strictly feasible. ✅
- `optimization_results.json` + `optimization_table.json` +
  `optimization_constraints.json` on disk end-to-end. ✅

### Phase 5a' — Edit-and-re-solve gate loop ✅
**Status:** complete. The `optimization` gate now supports a third
resolution alongside approve / reject: **rerun**. The user submits new
`run.options["optimization"]` overrides via `POST /runs/{id}/rerun`; the
runner consumes the payload, re-executes the optimization agent with
the merged options (overlaying onto whatever was passed at run-creation
time), and re-arms the gate for another review cycle. Loop exits on
approve (continue downstream) or reject (fail run).

**Backend**
- `core/orchestrator/gates.py` — `GateState.rerun_payload` channel;
  `gate_registry.request_rerun()` + `reset()`; `RERUNNABLE_AGENTS`
  whitelist (currently `{"optimization"}`).
- `core/orchestrator/runner.py` — `_wait_for_gate` is now a loop;
  consumes `rerun_payload`, merges into `run.options[agent_name]`,
  resets the agent's `AgentResult`, re-executes the agent, and
  re-arms the gate. Emits `agent_rerunning` events for SSE.
- `api/routes/approvals.py` — `POST /runs/{id}/rerun?agent=...` with
  a JSON body of constraint overrides. 400 for non-rerunnable agents;
  409 if the gate was already approved / rejected.

**Frontend**
- `web/lib/api.ts` — `rerunAgent(runId, agent, options)` helper.
  Constraint-editor UI lands in 5b alongside the validation card.

**Tests** (11 new; full suite 105 passed)
- `tests/unit/test_gate_rerun.py` — whitelist contains optimization;
  non-rerunnable agents get rejected; payload + event semantics; reset
  clears state; endpoint surface (200 / 400 / 409); end-to-end loop
  driven by `asyncio.create_task` that observes the agent re-running
  and final approval exiting the loop; option-merge layering; reject
  still works without firing a rerun.

### Phase 5b — Validation + constraint editor UI ✅
**Status:** complete. Rolling-origin CV-backed validation agent ships
end-to-end; the optimization AgentCard now exposes a recommendation
table + an inline constraint editor that exercises the 5a' rerun loop;
the validation AgentCard renders per-PPG verdicts with sign-stability /
WAPE / elasticity CV chips.

On the synthetic panel: 2/4 OLS-winning PPGs pass all checks, 2 fail
(one on `elasticity_cv` = 0.76, one on 75% sign-stability + CV = 1.51
— both real signals that the model's per-PPG elasticity wanders across
time windows even though point-WAPE looks fine).

**Backend**
- `core/validation/rolling.py` — expanding-window fold builder + per-fold
  refit of the winning OLS family. Returns elasticity, sign flag, R²,
  train + test WAPE per fold.
- `core/validation/checks.py` — `evaluate_ppg()` aggregates fold metrics
  into a pass / warn / fail verdict against four rules: sign stability
  ≥ 0.75 (pass) / 0.50 (warn); mean hold-out WAPE ≤ 0.20 / 0.30;
  elasticity CV ≤ 0.4 / 0.7; |mean ε| inside [0.3, 6.0].
- `core/agents/validation.py` — orchestrates per-PPG rolling CV, honours
  `run.options["validation"].n_folds` (default 4). Writes:
  - `validation_report.json` — full per-PPG verdict + per-fold detail +
    thresholds.
  - `validation_table.json` — flat one-row-per-PPG rows for the UI.
- `core/orchestrator/runner.py` — registers `ValidationAgent`.

**Frontend**
- `web/components/tables/RecommendationTable.tsx` — optimization output
  table: base + recommended price, %Δ chip (green up / amber down),
  units / revenue / margin, feasibility status (`feasible` /
  `relaxed`).
- `web/components/tables/ValidationTable.tsx` — validation verdict
  table: pass/warn/fail pill per PPG, sign stability %, mean WAPE, ε
  mean & CV, fold count.
- `web/components/ConstraintEditor.tsx` — inline form for objective,
  price ladder, margin floor, comp gap, max move; calls `rerunAgent()`
  from 5a' and the runner re-solves + re-arms the gate.
- `web/components/AgentVisuals.tsx` — new `OptimizationVisuals`
  (recommendation table + constraint editor) + `ValidationVisuals`
  (verdict table).
- `web/components/AgentCard.tsx` — `agent_rerunning` SSE event renders
  a "Re-solving with new constraints…" amber banner that suppresses
  the approve / reject buttons while the rerun is in flight.
- `web/lib/agent-meta.ts` — validation card chips: `n_validated`,
  `n_folds`, `n_pass`/`n_validated`, `n_fail`.

**Tests** (12 new; full suite 117 passed)
- `tests/unit/test_validation.py` — fold builder returns N folds with
  expanding train + non-overlapping test windows; empty when frame too
  short; train strictly precedes test in calendar order; per-fold
  refit recovers correct sign on a clean DGP; verdict aggregator emits
  pass / warn / fail under each rule (sign flip, high WAPE, high CV,
  no folds); validation agent writes both artefacts; LightGBM winners
  skipped; `run.options["validation"].n_folds` override respected.

**Verification**
- End-to-end on synthetic: validation agent runs after optimization,
  surfaces 2 stable PPGs and 2 unstable ones; `validation_report.json`
  + `validation_table.json` on disk.
- `pnpm build` clean; `/runs/[id]` first-load JS still around 322 kB
  raw (under the 350 kB gz target after compression).

**Acceptance gate**
- Holdout WAPE reported per PPG. ✅
- Rolling-origin CV with ≥3 folds per PPG. ✅
- Elasticity stability surfaced as CV across folds. ✅
- Sign-recovery rate surfaced as % of folds with correct sign. ✅
- Constraint editor wires into the rerun loop; UI shows re-solving
  state. ✅

## Phase 6 — Insights + Report + Polish
Insights agent, HTML + PDF report (jinja + weasyprint), cost dashboard, run replay, dark mode, error/retry states.

**Status:** in progress (6a shipped; UI polish backlog still open).

### Phase 6a — Insights agent + HTML/PDF report + cost dashboard ✅
**Status:** complete. The previously-stubbed `insights` agent now reads
every upstream artefact, builds a structured executive payload, and
renders the same content as both a self-contained HTML report and a
WeasyPrint-generated PDF. A per-run cost rollup (per-agent tokens,
USD, duration) lands on disk alongside the report and surfaces as a
collapsible dashboard at the bottom of the run page.

End-to-end on the synthetic panel: 4 PPGs optimised (3 feasible / 1
relaxed), 2/4 validation pass, ~$378k recommended revenue, report
PDF ≈ 35 kB, total pipeline ≈ 12 s wall-clock in dry-run mode.

**Backend**
- `core/report/builder.py` + `core/report/templates/report.html.j2` —
  Jinja env with currency / pct / signed-pct / num filters; A4-print
  CSS embedded in the template head; `build_html(payload)` →
  `build_pdf(html)` two-step pipeline. WeasyPrint import is deferred
  so the lightweight HTML path stays cheap.
- `core/llm/cost.py` — extends the existing per-call estimator with a
  run-level `summarise_run(run)` that rolls
  `tokens_in / tokens_out / cost_usd / duration` off every
  `AgentResult` into a typed `AgentCost[]` + `CostTotals`.
- `core/agents/insights.py` — `InsightsAgent`: indexes
  `optimization_table`, `validation_table`,
  `model_choice_summary`, `decomposition_table`, and
  `optimization_constraints` by PPG, asks the LLM for an exec headline
  + per-PPG rationales (dry-run fallback emits deterministic strings),
  writes `insights_summary.json`, `cost_summary.json`,
  `report.html`, and `report.pdf`. PDF write is wrapped — if WeasyPrint
  fails the agent still completes with `outputs.pdf=false`.
- `core/orchestrator/runner.py` — replaces the StubAgent for the
  `insights` stage with the real `InsightsAgent`.
- Adds `jinja2>=3.1.6` + `weasyprint>=68.1` to `pyproject.toml`.

**Frontend**
- `web/components/tables/InsightsSummary.tsx` — exec headline, four KPI
  tiles, HTML/PDF download buttons, per-PPG recommendation table with
  verdict pills + signed delta colouring.
- `web/components/CostDashboard.tsx` — collapsible per-agent table
  (tokens in/out, USD, duration) with a totals row; rendered between
  the run timeline and the artifact gallery. Shows a "dry-run, no
  tokens recorded" hint when no agent spent money.
- `web/components/AgentVisuals.tsx` + `AgentCard.tsx` — adds
  `insights` to the visuals + LLM-thinking allowlists; `AgentVisuals`
  now receives `agentState` so the insights panel can detect whether
  the PDF actually wrote and hide the download button on PDF failure.
- `web/components/RunTimeline.tsx` — wires `CostDashboard` in.
- `web/lib/agent-meta.ts` — insights card chips: PPGs reported,
  recommended revenue, recommended margin, HTML+PDF / HTML-only.

**Tests** (6 new; full unit suite 123 passed)
- `tests/unit/test_insights.py` — `build_html` renders every required
  section + signed-delta filter; `build_pdf` returns valid PDF bytes;
  `InsightsAgent` writes all four artefacts end-to-end with seeded
  upstream JSON; dry-run headline fallback fires when no API key;
  `summarise_run` rolls per-agent tokens + USD + duration correctly.

**Acceptance gate**
- HTML + PDF report renders cleanly from real upstream artefacts. ✅
- Cost dashboard sums per-agent tokens / cost / duration. ✅
- Insights agent succeeds even when WeasyPrint can't run (HTML still
  written, output flag surfaces the PDF failure). ✅

### Phase 6b — UI polish: shared table + executive banner ✅
**Status:** complete (first slice of the polish backlog). Adds a
generic `<ResultsTable>` (sortable headers, sticky first column,
row-level severity accent), refactors `RecommendationTable` and
`ValidationTable` onto it, drops an executive-summary banner at the
top of `/runs/[id]` that pulls headline + KPI tiles + report download
links from `insights_summary.json` once insights is done, and polishes
the runs index + the per-agent card statuses.

**Frontend**
- `web/components/tables/ResultsTable.tsx` — typed `ColumnDef<T>[]`
  contract with `format` / `sortValue` / `severity` / `numeric`;
  click-to-sort headers with `aria-sort`; sticky first column on
  horizontal scroll; optional 2px row-edge severity accent
  (pass/warn/fail/info/neutral); supports row click + highlight
  (for the modeling SHAP-row selection pattern).
- `web/components/tables/RecommendationTable.tsx` — refactored onto
  `ResultsTable`. Default sort: revenue desc. Adds row severity
  (relaxed → warn accent, feasible → pass).
- `web/components/tables/ValidationTable.tsx` — refactored. Default
  sort: verdict (fail first → pass last). Row severity = verdict.
- `web/components/ExecutiveBanner.tsx` — new top-of-page section.
  Only renders once `agents.insights.status === "done"`. Fetches
  `insights_summary.json`, shows headline + four KPI tiles (PPGs
  optimised, strict feasible, validation pass count, recommended
  revenue) with tone-coded borders (emerald / amber by health) +
  HTML / PDF download buttons. Hides if PDF didn't write.
- `web/components/RunTimeline.tsx` — wires the banner in between
  `RunHeader` and the agent timeline. Detects PDF availability via
  `agents.insights.artifacts` or `outputs.pdf` flag.
- `web/components/AgentCard.tsx` — card-level status colour: idle
  cards lose 30% opacity, running gets an amber edge, awaiting
  approval purple, failed rose; collapsed failed cards now surface a
  truncated error one-liner so reviewers don't have to expand.
- `web/app/runs/page.tsx` — runs index now renders a "+ New run"
  CTA, a relative timestamp (`5m ago` / `2h ago` / `3d ago`) per
  row, and a status-tone-coded pill. Errors from `listRuns()` surface
  as a rose-tinted banner instead of silently showing "No runs yet."

**Verification**
- `pnpm build` clean; `pnpm exec tsc --noEmit` clean.
- `/runs/[id]` first-load JS 325 kB raw (~90 kB gz, well under the
  350 kB gz target).
- Full pytest sweep: 123 passed (no Python changes).

**Acceptance**
- Shared `<ResultsTable>` used by ≥ 2 agent cards. ✅
- Headers click-to-sort; default sorts pick the most-useful column
  per table (revenue desc, verdict severity asc). ✅
- Executive banner appears at the top of the run page once insights
  is done, with one-click access to the HTML + PDF reports. ✅
- Runs index renders relative timestamps + clear status pill, plus an
  error state when the API is unreachable. ✅

### Phase 6c — Run-replay scrubber ✅
**Status:** complete (first slice). Adds a replay control at the top of
`/runs/[id]` that scrubs through the persisted `events.jsonl` stream.
The user can drag to any moment in the run, hit play to watch it
animate forward, or click "live" to snap back to the running tail. While
scrubbing, every agent card derives its status from filtered events
only — the live runState is ignored so the page faithfully shows what
the user-visible state actually was at that moment.

**Frontend**
- `web/components/ReplayBar.tsx` — slider with tick marks coloured by
  event type (sky `run_started`, amber `agent_started`, emerald
  `agent_finished`, rose `agent_failed`, purple `approval_required`),
  play / pause / live buttons, elapsed-time + clock-time labels, and a
  "frozen at HH:MM:SS · <agent> · <event>" status pill. Snaps to live
  when dragged to the trailing edge. Auto-paces playback to ~80 frames
  across the run's wall-clock span.
- `web/components/RunTimeline.tsx` — tracks `scrubTs` state; computes
  `visibleEvents` by filtering `events` against the cutoff; passes
  filtered events to every `AgentCard`; suppresses `runState` for
  cards / cost dashboard / artifact gallery during replay so derived
  status comes purely from event order; hides the executive banner
  during replay (it would otherwise show insights summary regardless
  of the scrub position).
- `web/components/AgentCard.tsx` — adds an event-derived
  `formatDuration` fallback (uses `agent_started` + `agent_finished`
  timestamps from the events stream) so per-card timing stays visible
  during replay even though `agentState.started_at` / `finished_at`
  aren't available.

**Verification**
- `pnpm build` + `tsc --noEmit` clean. `/runs/[id]` first-load JS 327 kB
  raw (~91 kB gz, well under the 350 kB gz target).
- SSE `events` endpoint replays history into the scrubber on page
  load (sanity-checked against an existing run in `runs/`).
- Python suite: 123 passed (no backend changes).

**Acceptance**
- Scrubber covers the entire run wall-clock span and snaps cleanly
  between scrubbed + live modes. ✅
- Tick density on the rail reflects the real event distribution
  (cluster of ticks where agents fire rapidly, gaps where one agent
  runs alone). ✅
- Each card's status, summary chips, and duration recompute from
  the visible event slice during replay. ✅

### Phase 6d — Remaining polish ✅
**Status:** complete. AnomalyTable + CandidatesTable now ride on the
shared `<ResultsTable>` (the latter via a new `expandable` config that
adds a leading ▸/▾ column and renders the per-PPG attempts block
inline). ReplayBar honours `space` (play/pause) and `←` / `→`
(step one event); the rail surfaces the shortcut hints on the right.
A collapsible run sidebar fetches recent runs and links between them
without leaving `/runs/[id]`. Five new charts wire onto the agent
cards.

**Frontend**
- `web/components/tables/ResultsTable.tsx` — adds an optional
  `expandable: { render, initialExpanded }` prop. When set, the table
  prepends a ▸/▾ column whose click toggles an expansion row spanning
  the full table width; controlled internally so callers stay
  declarative.
- `web/components/tables/AnomalyTable.tsx` — refactored onto
  `ResultsTable`. Default sort: severity ascending (error → info), row
  severity accent matches the pill colour. Empty-state message
  preserved.
- `web/components/tables/CandidatesTable.tsx` — refactored onto
  `ResultsTable`. Uses `expandable` for the per-PPG attempts pane,
  `highlightKey` for the SHAP-row selection, `onRowClick` for select.
  Default sort: test WAPE asc so the best fit floats to the top.
- `web/components/ReplayBar.tsx` — adds a keyboard listener: `Space`
  toggles play/pause (snapping to start if at the end), `ArrowLeft`
  / `ArrowRight` jump to the previous / next event tick. Keystrokes
  inside inputs / textareas / contenteditable are ignored. New `<kbd>`
  hint cluster on the right rail.
- `web/components/RunSidebar.tsx` — new collapsible left sidebar.
  Client-side `listRuns()` fetch, status-coloured dots, relative
  timestamp per row, active-run highlight, "+ new run" footer link.
  Collapses to a single ▸ button to keep the timeline full-width.
- `web/app/runs/[id]/page.tsx` — wraps `RunTimeline` in a 2-column
  flex layout so the sidebar sits beside the timeline on `lg+` and
  stacks above on small screens.
- `web/components/charts/FittedVsActual.tsx` — scatter of observed vs
  predicted units per PPG; train (green) and test (yellow) coloured,
  identity line dashed, Pearson r in the title.
- `web/components/charts/DecompStackedArea.tsx` — stacked area of
  `base + Σ due-by-group` per week per PPG, with the observed line
  overlaid; PPG selector inline.
- `web/components/charts/SimulationHeatmap.tsx` — price-multiplier ×
  promo grid coloured by revenue / margin / units (toggleable);
  PPG selector inline.
- `web/components/charts/ConstraintBinding.tsx` — horizontal bar of
  per-PPG chosen-cell slacks for every active constraint; negative
  bars mean the constraint was relaxed.
- `web/components/charts/ResidualHistogram.tsx` — 18-bin histogram
  of pooled hold-out residuals (`observed - predicted` on log-units)
  per PPG, with μ / σ / n in the title.
- `web/components/AgentVisuals.tsx` — wires the new charts:
  - modeling gets fitted-vs-actual under the selected SHAP PPG.
  - decomposition + simulation pick up first-class visuals (each
    with an inline PPG picker).
  - optimization gets a constraint-binding bar between
    recommendations and the editor (only when chosen_slacks is
    populated, which it now always is).
  - validation gets a residual histogram with a PPG picker.
- `web/components/AgentCard.tsx` — adds `decomposition` and
  `simulation` to the `VISUALS_AGENTS` allowlist so their cards open
  to the new visuals.

**Backend**
- `core/optimization/milp.py` — `MILPResult.chosen_slacks` (dict[str,
  float]) carries the chosen cell's per-constraint slack. Both
  strict + relaxed solve paths populate it; the relaxed path emits
  negative values for the constraints it had to break.
- `core/agents/optimization.py` — surfaces `chosen_slacks` on the
  per-PPG MILP block in `optimization_results.json`.
- `core/agents/modeling.py` — new `_collect_fitted_vs_actual()` runs
  the winning predictor on each PPG's full feature frame, writes
  `fitted_vs_actual.json` with parallel arrays of observed /
  predicted units (and log-units) + train/test split labels.
- `core/agents/validation.py` — new `_collect_residuals()` flattens
  fold-level hold-out residuals into a per-PPG bag, writes
  `validation_residuals.json`.
- `core/validation/rolling.py` — `fit_one_fold()` now records
  `test_residuals_log` on each fold so the agent has the raw
  residuals to roll up.
- `core/models/{loglog_ols,semilog_ols,lightgbm_model}.py` — each
  fitter stashes `diagnostics["test_residuals_log"]` whenever a test
  frame is provided (parallel to `test_wape`).

**Tests** (extended existing tests; full unit suite 138 passed)
- `tests/unit/test_modeling.py` — asserts `fitted_vs_actual.json`
  contains a row per fit PPG with parallel observed / predicted
  arrays and a `n_train + n_test = n` split.
- `tests/unit/test_validation.py` — asserts `validation_residuals.json`
  is written, has per-PPG residual lists whose lengths sum across
  folds, and contains float values.
- `tests/unit/test_optimization.py` — extends the strict + relaxed
  MILP tests to assert `chosen_slacks` is populated with the active
  constraints (>= 0 when feasible; at least one < 0 when relaxed).

**Verification**
- `pnpm build` clean; `pnpm exec tsc --noEmit` clean.
- `/runs/[id]` first-load JS 338 kB raw (~95 kB gz, under the 350 kB
  target).
- End-to-end on synthetic: `fitted_vs_actual.json` (8 rows),
  `validation_residuals.json` (8 rows), `chosen_slacks` on every MILP
  cell. Modeling, decomposition, simulation, optimization, validation
  cards all surface the new visuals.
- Full unit suite green (138 passing).

**Acceptance**
- AnomalyTable + CandidatesTable share `<ResultsTable>` (expandable
  rows work). ✅
- Space + ←/→ keyboard shortcuts move the replay scrubber. ✅
- Cross-run sidebar surfaces every recent run from any
  `/runs/[id]`. ✅
- Each of the five "more graphs" backlog items renders on its
  respective agent card. ✅

---

## Post-review fixes

### P1 — Correctness & contract ✅
**Status:** complete.
- `core/data/expectations.py` — added `panel_drift_suite` and `panel_anomaly_suite`
  (completing all five suites described in the architecture docs); drift suite
  reads `core/data/baselines/synthetic.json` and skips gracefully when absent.
- `core/orchestrator/gates.py` — `reset()` now calls `state.event.clear()`
  instead of replacing the Event, eliminating the race where a concurrent
  awaiter on the old object would be orphaned.
- `web/components/charts/FittedVsActual.tsx` — Pearson `r` is now computed on
  `observed_log` / `predicted_log` (log space) to match `test_wape`; labelled
  `r (log)` so the scale is explicit.
- `web/components/ReplayBar.tsx` — right-arrow at end-of-stream now clamps to
  `tLast` instead of jumping to live; user must click the "live" button
  explicitly.

---

## Datasets & published-elasticity benchmarks ✅
**Status:** complete. Two largely independent additions on top of the
shipped phases — neither blocks any phase plan, both close gaps the
client-UI prototype surfaced ("can we run on real data?", "how do our
elasticities compare to the literature?").

### Workstream A — Dominick's Finer Foods loader ✅
**Status:** complete. The pipeline now accepts the Kilts Center's
Dominick's scanner panel as a first-class data source alongside the
synthetic generator. End-to-end on a downloaded category set:
`automl prepare-dominicks --categories yogurt,beer --out data/dominicks.csv`
→ `automl run --data data/dominicks.csv --no-gates` runs the full DAG
with no agent code touched past ingestion.

**Backend**
- `core/data/loaders/dominicks.py` — converts per-category movement
  (`w<code>.csv`) + UPC dictionary (`upc<code>.csv`) CSVs into the
  canonical panel schema:
  - week → date via the published Dominick's anchor (week 1 starts
    Thursday 1989-09-14).
  - `price = PRICE / max(QTY, 1)` (per-unit shelf price; e.g. 6-pack
    beer at $5.99 becomes ~$1.00 per unit).
  - `base_price` = trailing 13-week max of price on non-promo weeks
    per (sku, store), clamped to be ≥ live price.
  - `tpr_flag = 1` when `SALE ∈ {B,S,C}` (bonus buy / sale / coupon).
  - `display_flag` / `feature_flag` = 0 (not recorded in Dominick's).
  - `distribution_acv` = 100 (single chain).
  - `region` = "Chicago", `competitor_price` = null.
  - Brand inferred from the first all-caps token of `DESCRIP`;
    `pack_size` from `SIZE`; `segment` from `COM_CODE`.
  - `holiday` tagged from the `holidays` package (US federal) if the
    week_start falls within 7 days of a holiday.
- `core/data/loaders/dominicks_categories.py` — Dominick's 4-letter
  category code → (label, display name, benchmark key) map. Covers
  27 categories spanning Hoch's 18 + a handful of Dominick's-later
  additions (frozen dinners, paper towels, bathroom tissue, oatmeal,
  yogurt).
- `cli/run.py` — registers `automl prepare-dominicks` as a Typer
  subcommand: reads `data/dominicks-raw/`, runs the loader for the
  requested categories, validates the output against
  `REQUIRED_COLUMNS`, writes the panel CSV.

**Constraints**
- Kilts data-use agreement forbids redistribution → raw archive stays
  local under `data/dominicks-raw/` (added to `.gitignore` with the
  anchored `/data/dominicks-raw/` pattern).
- Output `data/dominicks.csv` already matched by the existing
  `data/*.csv` gitignore rule.

**Tests** (5 new; full unit suite 167 passed)
- `tests/unit/test_dominicks_loader.py` — fixture-based round trip on
  a hand-crafted yogurt + beer mini-archive: week→date anchor,
  schema compliance, OK=0 row filtering, per-unit price derivation,
  promo / `base_price` relationship, `PanelRow` pydantic validation,
  unknown-category error, empty-archive error.

**Acceptance**
- `automl prepare-dominicks --help` lists every option. ✅
- Output CSV passes `validate_panel`. ✅
- Output CSV ingests cleanly via `automl run --data <csv>` (manually
  verified on a downloaded yogurt + beer subset). ✅
- Raw archive is gitignored; loader test fixtures live in
  `tests/unit/test_dominicks_loader.py` (no real Dominick's bytes
  committed). ✅

### Workstream B — Published elasticity benchmarks (Hoch 1995 + Bijmolt 2005) ✅
**Status:** complete. The validation agent now compares each PPG's
recovered elasticity against the published category range from the
two canonical CPG elasticity references, surfacing the result on
every row of `validation_table.json` and as a 7th gate card on the
client-facing validation page.

**Backend**
- `core/benchmarks/data/elasticity.json` — static table (25 categories):
  - Hoch, Kim, Montgomery & Rossi (1995, JMR 32:1) — 18 Dominick's
    categories with chain-level elasticities (beer −1.32, soft drinks
    −3.18, cookies −3.96, cheese −3.27, …).
  - Bijmolt, van Heerde & Pieters (2005, JMR 42:2) — grand mean −2.62
    used as the global fallback; category-level rows for yogurt, salty
    snacks, ice cream, coffee, paper products that Hoch didn't cover.
  - Per-category band: `lo / mean / hi` (mean ± max(0.5, 0.3·|mean|),
    approximating a ±2·SE envelope around the Hoch point estimates).
  - Alias map ("Soda" → soft_drinks, "Juice" → bottled_juice, "Frozen
    pizza" → frozen_entrees, …) so the synthetic generator's labels
    and the Dominick's loader's labels both join cleanly.
- `core/benchmarks/elasticity.py` — `lookup_category(category)` does an
  exact / alias / substring match; `classify(elasticity, bench)`
  returns one of `in_band` / `out_band_low` (less elastic than band) /
  `out_band_high` (more elastic than band) / `no_benchmark` (NaN-safe).
- `core/agents/validation.py` — loads `ppg_mapping_table.json` to
  resolve PPG → category, augments every row of
  `validation_table.json` with `benchmark_status`, `benchmark_mean`,
  `benchmark_low`, `benchmark_high`, `benchmark_source`,
  `benchmark_category`. Adds `n_in_benchmark` / `n_out_benchmark` /
  `n_no_benchmark` / `benchmark_pass_rate` to `result.outputs`. The
  existing sign / WAPE / CV / magnitude verdict is unchanged.

**Frontend** (client-facing validation page, `web-client/`)
- `web-client/lib/types.ts` — `PPGForestPoint` gains optional
  `benchmark_low / benchmark_high / benchmark_mean / benchmark_source
  / benchmark_status` fields.
- `web-client/lib/mock.ts` — mock forest populates the new fields per
  category; `buildValidation()` now emits a 7th gate card
  ("Benchmark alignment") reporting in-band count vs total.
- `web-client/components/charts/ConfidenceForest.tsx` — dashed
  reference line at the Bijmolt grand mean (−2.62) with a label;
  warning-coloured stroke on any bar whose elasticity sits outside
  the published band; tooltip shows the published band alongside the
  point estimate.
- `web-client/app/validation/page.tsx` — header copy updated for the
  7-gate count and the literature reference; forest subtitle now
  calls out the benchmark band.

**Tests** (15 new; full unit suite 167 passed)
- `tests/unit/test_elasticity_benchmarks.py` — table loads with the
  Bijmolt grand mean and a representative set of categories; exact
  key lookup hits; alias lookup resolves Soda/Juice; unknown
  category and `None` return `None`; `classify` returns
  in_band / out_band_high / out_band_low / no_benchmark / NaN.
- `tests/unit/test_validation_benchmark_check.py` —
  `_load_ppg_categories` reads ppg_mapping_table.json correctly;
  missing-mapping graceful empty; per-PPG classify hits expected
  status (in_band yogurt, out_band beer at -10, no_benchmark
  Antimatter); summary counts roll up correctly across PPGs.

**Acceptance**
- `validation_table.json` rows carry every `benchmark_*` field. ✅
- `result.outputs` exposes `benchmark_pass_rate` without removing
  any existing key (back-compat). ✅
- Client validation page renders 7 gate cards (was 6). ✅
- Forest chart shows the Bijmolt grand-mean reference line and
  warning stroke on out-of-band PPGs. ✅
- Web-client `tsc --noEmit` + `next build` clean. ✅
- Full unit suite (167 tests) green. ✅
