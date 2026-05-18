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
Make every Phase-1/2 agent's work visible. A reviewer must be able to *see* what
the data looks like, which quality rules fired and why, how SKUs got grouped,
and which feature decisions got made — not just whether each agent returned
`done`. Decided: inline layout (each AgentCard expands with its own visuals),
Apache ECharts for chart rendering. Mockups under `web/app/dev/` show the
target visual style.

### Backend — chart-ready artefacts
Each agent gets a follow-on JSON the frontend can render without re-querying
DuckDB. New module `core/data/charts.py` builds them; agents write them
alongside their existing artefacts.

| Agent | Adds |
|---|---|
| ingestion | `coverage_grid.json` (sparse SKU × week matrix), `weekly_trend.json` (panel-wide units + price + promo share); enrich `ingestion_findings.json` with severity + row counts |
| ppg_mapping | `ppg_scatter.json` (per-SKU x = price tier, y = log price, colour = PPG), `ppg_price_box.json` (per-PPG min/q1/median/q3/max) |
| ppg_selection | per-metric eligibility breakdown for a stacked-bar view |
| eda | enrich existing `eda_report.json` with chart-ready arrays (already has target_relationship + pairwise_corr) |
| feature_engineering | `feature_histograms.json` (20-bin histograms + mean/std per engineered column) |
| feature_refine | enrich existing `feature_refine.json` with the refined-set correlation matrix |

Every LLM-using agent also writes `<agent>_llm_trace.json` capturing system
prompt, user prompt, raw response, and whether the dry-run fallback fired —
so the "agentic layer" is auditable, not opaque.

### Frontend — ECharts integration (inline)
- Add `echarts` + `echarts-for-react`, tree-shaken to `core` + the chart /
  component modules we use; gz cost target < 200 kB.
- `web/components/charts/EChart.tsx` — `"use client"` wrapper that
  `dynamic({ ssr: false })`-imports echarts, applies a slate dark theme,
  exposes a typed `option` prop.
- One chart component per type: `TrendChart`, `CoverageHeatmap`, `PPGScatter`,
  `PPGPriceBox`, `CorrHeatmap`, `VIFBar`, `FeatureHistograms`.
- Plain-React tables stay: `DataPreview`, `SchemaTable`, `QualityPanel`,
  `AnomalyTable`, `DropLog`.

Inline integration in `AgentCard.tsx` (one new section per agent, shown when
status is `done` or `awaiting_approval` and the artefact exists):

- **ingestion** → DataPreview + SchemaTable + CoverageHeatmap + QualityPanel + AnomalyTable
- **ppg_mapping** → PPGScatter + PPGPriceBox (and pulls the existing PPGTable into the card body)
- **ppg_selection** → eligibility stacked-bar
- **eda** → TrendChart + target-relationship table + CorrHeatmap
- **feature_engineering** → FeatureHistograms grid
- **feature_refine** → VIFBar + refined CorrHeatmap + DropLog

A new "Agent thinking" sub-section in every expanded card surfaces the
LLM trace (with a `dry-run` badge when the fallback ran) so reviewers can see
exactly what prompt produced the rationale on screen.

### Tests
- `tests/unit/test_charts.py` — every chart-data tool runs against the
  synthetic warehouse and returns non-empty, well-formed shapes.
- `tests/unit/test_artifact_completeness.py` — end-to-end run; assert every
  Phase 1/2 agent emits the expected chart-ready artefact under the run dir.
- Playwright `tests/e2e/test_run_page_visuals.spec.ts` — boot API, drive a
  no-gate run, expand the ingestion / ppg_mapping / eda / feature_refine
  cards, assert each chart container renders with non-zero dimensions and no
  console errors.

### Acceptance gate
- **Coverage:** every Phase 1 + Phase 2 agent has at least one chart / table
  visible inside its expanded card.
- **Quality story is visible:** Ingestion card surfaces dbt + GE rules with
  pass / warn / fail pills, the rule's message, row count, and severity. The
  anomaly table lists every flagged row with the reason.
- **PPG rationale is visible:** the SKU scatter + price-box together let a
  reviewer accept the grouping at a glance.
- **Feature decisions are visible:** VIF bar + drop log + refined corr heatmap.
- **Agentic layer is auditable:** every LLM-using agent surfaces its prompt /
  response or marks itself dry-run.
- **Performance:** `/runs/[id]` first-load JS stays below ~350 kB gz.
- **Tests green:** existing 7 pytest + new chart-data tests + Playwright snapshot.

### Implementation order
1. `core/data/charts.py` + chart-ready artefacts from each agent.
2. `tests/unit/test_charts.py` + `test_artifact_completeness.py`.
3. ECharts install + `EChart` wrapper + chart components.
4. Inline integration in `AgentCard.tsx`; remove the now-redundant standalone PPGTable below the timeline.
5. LLM trace artefact + "Agent thinking" sub-section.
6. Playwright visual snapshot test.
7. Verification screenshots on the PR; flip Phase 2a to ✅.

**Status:** planned. Mockups under `web/app/dev/` (inline / tabs / subroutes) for design comparison; inline + ECharts chosen.

## Phase 3 — Modeling + Results Reasoning
Model tools (log-log, semi-log, LightGBM+SHAP, PyMC hierarchical), iterative modeling agent that retries on wrong-sign elasticities, results-reasoning agent, model-choice approval gate, elasticity chart UI. Verify log-log recovers truth on synthetic.

**Status:** pending.

## Phase 4 — Decomposition + Simulation
Decomp + sim tools, two agents, stacked-bar (due-to) + scenario-grid heatmap UI. Verify decomposition reconciles to observed.

**Status:** pending.

## Phase 5 — Optimization + Validation
Opt tools, constraint-elicitation gate, scipy continuous warm start → PuLP MILP with ladder/margin-floor/comp-gap, validation agent (holdout WAPE, elasticity reasonableness, stability), constraint editor + recommendation table UI.

**Status:** pending.

## Phase 6 — Insights + Report + Polish
Insights agent, HTML + PDF report (jinja + weasyprint), cost dashboard, run replay, dark mode, error/retry states.

**Status:** pending.
