# Stage FAQs & Corner Cases

Per-stage reference for the 14-agent CPG price-and-promo pipeline.

Each section is split into:

- **Client questions** — what an analyst running the platform is most likely to
  ask, with plain-English answers.
- **Corner cases** — edge inputs and what the pipeline does in each one.
- **Engineering notes** — implementation pointers (file paths, env knobs,
  defaults).

Anchors are stable and linked from the `Common questions & corner cases`
section on each agent card in the web UI (`web/components/AgentFAQ.tsx`,
`web/lib/agent-faqs.ts` is the source of truth — keep this doc in sync).

## Contents

1. [Ingestion](#ingestion)
2. [PPG Mapping](#ppg-mapping)
3. [PPG Selection](#ppg-selection)
4. [Feature Selection](#feature-selection)
5. [EDA](#eda)
6. [Feature Engineering](#feature-engineering)
7. [Feature Refine](#feature-refine)
8. [Modeling](#modeling)
9. [Results Reasoning](#results-reasoning)
10. [Decomposition](#decomposition)
11. [Simulation](#simulation)
12. [Optimization](#optimization)
13. [Validation](#validation)
14. [Insights](#insights)

---

## Ingestion

Load the CSV into DuckDB, build the dbt panel mart, run schema + distribution
checks, profile each numeric column for outliers and drift.

### Client questions

- **Q: Why did the run fail before any models were fit?**
  A: Ingestion enforces minimums: ≥500 rows, ≥5 unique SKUs, ≥1 store. Below
  any floor the run halts. Widen the panel (more weeks or SKUs) and retry.

- **Q: My price column failed validation but the numbers look fine.**
  A: Price mean must fall in $0.50–$50.00 and stdev in $0.00–$20.00. Common
  cause: prices in cents rather than dollars, or a single SKU with an extreme
  outlier price.

- **Q: Why are my units flagged as outliers when they're real?**
  A: Q50 caps at 10k units, Q95 at 50k. Hypermarket-scale SKUs trip the warn.
  Either confirm and ignore, or split the file by channel.

- **Q: What does the drift warning mean?**
  A: Each numeric column's mean and Q25/Q75 are compared against a baseline
  snapshot. Drift outside ±40% triggers a warn. First runs have no baseline so
  drift is skipped.

- **Q: Why is base_price flagged when it's lower than transaction price?**
  A: Base price is the regular shelf price; transaction price is what shoppers
  paid (including promo). Base must be ≥ transaction in ≥99% of weeks.
  Reversed columns trip this.

- **Q: Can I run without distribution_acv?**
  A: Yes — the column is optional. If present it must be 0–100. If missing,
  the distribution coefficient simply drops out of the models.

- **Q: What if I have negative units (returns)?**
  A: Hard fail. Aggregate net units at the same grain or remove return rows
  upstream — the panel mart assumes non-negative weekly units.

- **Q: Why did the LLM narrative differ across two runs of the same file?**
  A: Only the narrative is LLM-generated. Thresholds and verdicts are
  deterministic. With no API key (or on non-JSON output) the narrative falls
  back to a template that is identical across runs.

### Corner cases

- **All weeks promo (TPR=1 everywhere)** → ingestion passes; flagged
  downstream at PPG eligibility, not here. User sees eligibility score low at
  PPG selection.
- **Single-store dataset** → ingestion passes (≥1 store satisfied); ACV
  becomes a constant. User sees the distribution coefficient identified as
  singular at modeling.
- **Missing weeks (date gaps)** → ingestion passes; EDA surfaces the coverage
  shortfall. User sees the coverage component pull PPG eligibility down.
- **Mixed currencies in one file** → not detected directly; price-distribution
  warn may fire if range straddles $0.50–$50. User sees a wide price-stdev
  warning.
- **Columns in mixed case** (`SKU` vs `sku`) → dbt source normalises to
  lower-case; both resolve.
- **Zero-variance price column** → ingestion passes; price_cv=0 marks every
  PPG ineligible at stage 3.
- **Upload exceeds `MAX_UPLOAD_MB`** (default 200) → API rejects upload with
  HTTP 413 before ingestion runs. User sees upload error; raise
  `MAX_UPLOAD_MB` env to allow.
- **Non-UTF8 CSV** (e.g. Windows-1252) → DuckDB read fails before checks run;
  surfaced as ingestion error.

### Engineering notes

- Threshold definitions: `core/data/expectations.py:18-107`
- Drift slack (`DRIFT_SLACK_PCT` env): `core/config.py` `drift_slack_pct`
- Baseline path (`BASELINE_DIR` env): `core/config.py` `baseline_dir`
- LLM dry-run fallback: `core/agents/ingestion.py` `_dry_run_*`
- Required columns: `sku, store_id, week_start, units, price, tpr_flag`

---

## PPG Mapping

Group SKUs into Price-Pack Groups by brand, category and price tier. Score
each mapping by within-group price coherence.

### Client questions

- **Q: Why are these two SKUs in different PPGs when they're the same product?**
  A: Brand and category form a hard partition — different brand or category
  SKUs never merge. Check those columns for spelling or case mismatches.

- **Q: Why didn't the premium tier get its own PPG?**
  A: Splitting needs at least 4 SKUs in the brand-category and a price spread
  of ≥0.50 log-dollars (~65% step), plus silhouette score ≥0.70. Narrow tiers
  stay merged.

- **Q: What does the confidence score on a SKU mean?**
  A: 1 − (distance-to-own-centroid / distance-to-nearest-other-centroid),
  clipped to [0,1]. Single-cluster default is 0.92.

- **Q: Can I edit the PPGs manually?**
  A: Not directly today — but the approval gate after this stage lets you
  reject and rerun ingestion with cleaned inputs.

- **Q: Why is one PPG flagged but still eligible?**
  A: A flag means within-group price ratio exceeds 2×. The cluster is
  internally heterogeneous; we keep it but surface for review.

- **Q: The LLM rationale changed between runs. Is the mapping unstable?**
  A: No. Clustering is fully deterministic (k-means with fixed seed). Only the
  narrative text varies if the LLM is in API mode.

- **Q: Should I rerun if I add new SKUs?**
  A: Yes — full reclustering happens, and PPG IDs may shift. Save prior
  recommendations before rerunning.

- **Q: Does seasonality affect PPG mapping?**
  A: No. PPGs are static across the whole period; seasonality enters at the
  modeling stage as week-of-year features.

### Corner cases

- **Single SKU in a brand/category** → single-SKU PPG with confidence 0.92.
- **2–3 SKUs in a brand/category** → single PPG (below the 4-SKU k-means
  floor).
- **All SKUs at identical price** → single PPG (log-price spread <0.50 below
  split threshold).
- **Silhouette score below 0.70 after attempted split** → split rejected;
  cluster kept whole.
- **New SKU added between runs** → full re-clustering; prior PPG IDs may
  shift. Recommendation table is keyed by new PPG IDs.
- **Same pack size, different brands** → never merged (brand partition is
  hard).
- **Missing brand or category column** → ingestion catches it first; never
  reaches PPG mapping.
- **Within-group price ratio >2×** → PPG kept but `flagged=true`; surfaced in
  eligibility narrative.

### Engineering notes

- Clustering rules: `core/ppg/cluster.py:33-98`
- Confidence formula: `core/ppg/cluster.py`
- LLM dry-run rationales: `core/agents/ppg_mapping.py` `_dry_run_*`
- Approval gate: `core/orchestrator/state.py` `APPROVAL_GATES`

---

## PPG Selection

Score each PPG on size, coverage, price variation and promo activity. Flag
the ones eligible for modelling.

### Client questions

- **Q: Why is my best-selling PPG marked ineligible?**
  A: Composite score below 0.60. Inspect components: volume, coverage, price
  CV, and promo %. Any one dragging hard can sink the total.

- **Q: What threshold makes a PPG eligible?**
  A: Overall score ≥ 0.60. Weights: volume 25%, coverage 25%, price CV 30%,
  promo 20%. Threshold is 0.60 inclusive.

- **Q: My PPG has only 60% week coverage — is that enough?**
  A: Coverage component scores below 0.60 there, but other components can
  offset. Always check the composite score, not a single component.

- **Q: We never promote. Can I still model?**
  A: Promo % expected in [5%, 50%]. Below 5% lowers the promo component to
  near zero. Below ~3% the PPG usually falls under the 0.60 threshold.

- **Q: Why is a PPG with massive volume scored low?**
  A: Volume saturates at 50k units. Beyond that you get full 0.25 credit but
  nothing extra — other components must still clear their bands.

- **Q: Can I lower the eligibility threshold?**
  A: Not via UI yet. Edit `core/ppg/score.py` defaults or override in code.
  Future enhancement: per-run override via approval gate.

- **Q: What is price CV?**
  A: Coefficient of variation (stdev / mean) of price across observed weeks.
  We saturate at 0.30; below 0.12 the score drops sharply.

- **Q: Do eligibility scores update when I edit constraints?**
  A: No. Eligibility is tied to the input data, not the optimization
  constraints. Editing constraints only re-solves the optimization stage.

### Corner cases

- **Zero price variation (price_cv=0)** → price-CV component pulls overall
  score to ~0.40; PPG marked ineligible.
- **100% promo weeks (TPR always on)** → promo % outside [5%,50%] band;
  score drops; usually ineligible.
- **Fewer than 5 weeks of data** → coverage near zero; ineligible.
- **Volume in millions of units** → saturates at 50k units (full credit); no
  penalty.
- **Very high price CV (>0.5)** → saturates at 0.30 (full credit); not
  penalised for being high.
- **All PPGs ineligible** → modeling skips every PPG; downstream stages
  cascade with empty results. User sees a red banner in insights.
- **LLM unreachable** → per-PPG rationales fall back to deterministic
  templates.
- **Score exactly 0.60** → eligible (inclusive lower bound).

### Engineering notes

- Scoring formula: `core/ppg/score.py:27-106`
- Threshold (0.60): `core/ppg/score.py` `ELIGIBILITY_THRESHOLD`
- Component weights: volume 25, coverage 25, price_cv 30, promo 20

---

## Feature Selection

Inspect panel columns; classify as target / identifier / temporal / flag /
numeric / categorical.

### Client questions

- **Q: Why was log_units picked as the target?**
  A: Heuristic match on canonical column names (`units`, `sales`, `qty`) plus
  numeric type. We then log-transform inside engineering.

- **Q: Which columns are classified as identifier?**
  A: `sku`, `store_id`, `ppg_id`, and any column with an `_id` suffix or
  constant within row.

- **Q: Why is `week_start` flagged as the temporal column?**
  A: Datetime-parseable plus name match. If parsing fails, the column drops
  back to numeric/categorical.

- **Q: Are categorical columns kept?**
  A: Yes, passed through to feature engineering. Low-cardinality categoricals
  get dummy-encoded; high-cardinality ones are dropped.

- **Q: Can I override the classification?**
  A: Not via UI today. Pre-clean your file or extend
  `core/agents/feature_selection.py`.

- **Q: Why is a 0/1 column treated as numeric?**
  A: Only columns suffixed `_flag` or with exactly 2 distinct values 0/1 are
  flagged. Other binary integers are kept numeric.

- **Q: What if my target is dollars not units?**
  A: Pipeline still runs; elasticity will then be a revenue elasticity (often
  closer to 1.0 in magnitude). Interpret accordingly.

- **Q: Why so few columns selected as numeric?**
  A: High-cardinality strings and date columns are filtered out. Recheck for
  stringified numbers (e.g. price stored as text).

### Corner cases

- **No clear target column** → stage fails with `target column not detected`;
  rename your sales/units column.
- **Two candidates for target** (e.g. both `units` and `sales`) → first by
  configured priority wins; the other becomes a numeric feature.
- **Date column with non-ISO format** → may parse as identifier; downstream
  EDA breaks with `temporal column unparseable`.
- **Boolean columns stored as 'true'/'false' strings** → not detected as
  flag; treated as categorical.
- **Numeric column with <2 unique values** → classified as flag if values are
  0/1, otherwise dropped.
- **All-null columns** → excluded silently; not reported as candidates.
- **Mixed dtype within a column** → pandas coerces to object; usually
  misclassified as categorical.
- **LLM unreachable** → deterministic template narrative; classification
  unchanged.

### Engineering notes

- Classification logic: `core/agents/feature_selection.py`
- Heuristic priorities: target → flag → numeric → categorical

---

## EDA

Roll up to PPG × week; compute numeric summaries, target relationships,
pairwise correlations, missingness.

### Client questions

- **Q: Why is the Spearman correlation NaN for some features?**
  A: Spearman requires at least 30 non-null paired observations. Sparse
  columns drop out as NaN.

- **Q: What threshold marks a relationship as 'strong'?**
  A: Display only: |ρ| ≥ 0.6 strong, ≥ 0.3 moderate. These are visual cues,
  not gates.

- **Q: Why is missingness high for `lag_units_4`?**
  A: The first 4 weeks of each PPG's series produce NaN lags by construction.
  Expected.

- **Q: Are correlations computed within PPG or across the panel?**
  A: PPG × week panel, pooled across PPGs. Within-PPG views are available in
  downstream visuals.

- **Q: Pearson vs Spearman — which should I trust?**
  A: Spearman is primary because elasticity relationships are non-linear in
  raw units. Pearson is reported for reference.

- **Q: What does 'candidates ranked' mean?**
  A: Features sorted by absolute Spearman ρ against the target. Top of the
  list = strongest monotonic signal.

- **Q: Why isn't price in the top correlates?**
  A: Price-unit relationship is most cleanly seen in log-log space. The raw
  correlation can be modest while the log elasticity is sharp.

- **Q: Why does EDA show fewer features than engineering produces?**
  A: EDA ranks candidates only. Engineering keeps the full feature set;
  pruning happens later in refine.

### Corner cases

- **Single PPG** → all cross-PPG variance collapses; correlations inflate.
  User sees a 'single-PPG run' banner.
- **Zero-variance numeric column** → excluded from correlation table.
- **All-null column** → excluded; not surfaced.
- **n = 29 paired observations** → Spearman returns NaN (one below the 30-row
  floor).
- **Categorical with high cardinality** → sampled to top-k levels; full
  distribution not shown.
- **Mostly-zero feature** (e.g. `competitor_price` 90% null) → low n after
  dropping nulls; usually NaN ρ.
- **Multi-collinear cluster of features** → all show similar high ρ; pruning
  lands in feature_refine.
- **LLM unreachable** → insights paragraph empty; tables still render.

### Engineering notes

- Spearman implementation: `core/agents/eda.py`
- Minimum n threshold: n ≥ 30 for ρ
- Display thresholds (UI only):
  `web/components/tables/TargetRelationship.tsx`

---

## Feature Engineering

Build lagged, holiday and competitive-price features at PPG × week grain.

### Client questions

- **Q: What features get built?**
  A: `log_price`, `log_units`, promo and price lags at lag-1 and lag-4,
  week-of-year seasonality dummies, and `distribution_acv` passthrough.

- **Q: Why lag 1 and lag 4 specifically?**
  A: Lag 1 captures last-week carryover; lag 4 covers a 4-week consumption
  cycle common in CPG. Hard-coded today; configurable later.

- **Q: Why is parquet sometimes CSV?**
  A: pyarrow is optional. If unavailable, we transparently fall back to CSV.
  Downstream readers handle either format.

- **Q: Are competitor prices included?**
  A: Yes if the column is present in your CSV. The join is left-outer and
  nullable; missing competitor data leaves the feature null.

- **Q: How are missing lag values handled?**
  A: First weeks per PPG produce NaN lags. The modeling stage drops those
  rows before fitting.

- **Q: Are holidays included?**
  A: Implicitly via week-of-year dummies. Explicit holiday flags
  (Thanksgiving, Easter) are on the roadmap.

- **Q: Can I add custom features?**
  A: Not via UI. Subclass `FeatureEngineering` or fork the agent.

- **Q: Why is `base_price` log-transformed but `distribution_acv` linear?**
  A: Prices follow log-log elasticity; shares are bounded [0,1] and naturally
  linear. This is the modelling convention.

### Corner cases

- **Series shorter than max lag (4 weeks)** → all lag values NaN; modeling
  stage skips that PPG.
- **Promo never active for a PPG** → promo lags all zero; coefficient
  unidentified during fit.
- **log(0) units** → units below 1 are floored to 1 before the log transform.
- **Parquet write fails (pyarrow missing)** → CSV fallback; downstream
  readers handle either path.
- **Single-store ACV (constant)** → coefficient unidentified at fit time;
  reported as singular.
- **`distribution_acv` missing entirely** → column dropped; no distribution
  coefficient produced.
- **Two-week file** → almost everything NaN; modeling skips most PPGs.
- **Year boundary crossing in week-of-year** → handled with modular (cyclic)
  dummies; no off-by-one.

### Engineering notes

- Feature list: `core/agents/feature_engineering.py` `ENGINEERED_COLUMNS`
- Lag values: lag 1, lag 4 (hard-coded)
- Parquet fallback: auto-detects pyarrow; CSV otherwise

---

## Feature Refine

Drop collinear features via VIF and absolute-correlation pruning. `log_price`
is protected.

### Client questions

- **Q: Why was my favourite feature dropped?**
  A: Either VIF > 10 (multicollinear with the rest) or |corr| > 0.95 with a
  peer. Both indicate redundancy.

- **Q: What is VIF?**
  A: Variance Inflation Factor: how much a feature's variance is inflated by
  collinearity with other features. >10 is the conventional pruning
  threshold.

- **Q: Is log_price always kept?**
  A: Yes. `log_price` is protected — it's the elasticity predictor and never
  gets dropped, even if its VIF is high.

- **Q: Why are the kept and dropped lists empty?**
  A: No engineered features survived earlier stages. Modeling will likely run
  on `log_price` only or skip.

- **Q: What's the |corr| threshold?**
  A: 0.95 absolute Pearson correlation. Pairs above this threshold have one
  member dropped (lower-VIF survives).

- **Q: Does refining happen per-PPG or pooled?**
  A: Pooled across the whole panel. Every modeled PPG sees the same refined
  feature set.

- **Q: What's the difference between feature_selection and feature_refine?**
  A: Selection picks raw columns to consider. Refine prunes the engineered
  set after lags/interactions are built. Two different stages.

- **Q: Can I override the kept set?**
  A: Not via UI today. Edit `core/agents/feature_refine.py` to add protected
  features.

### Corner cases

- **Fewer than 2 features survive refine** → modeling falls back to log-log
  on price alone.
- **`log_price` has high VIF on its own** → still kept (protected); the VIF
  warning surfaces but does not drop it.
- **All lag features perfectly correlated** → most dropped; one
  representative survives.
- **Singular matrix during VIF computation** → feature flagged suspect and
  dropped.
- **Promo features near-zero variance** → dropped silently as
  non-informative.
- **Single PPG in run** → any PPG-level dummies trivially dropped.
- **Engineered frame missing target column** → stage fails with explicit
  error; do not proceed to modeling.
- **LLM narrative differs across runs** → expected. The kept/dropped lists
  themselves are deterministic.

### Engineering notes

- VIF threshold: VIF < 10
- Correlation threshold: |corr| ≤ 0.95
- Protected features: `log_price` (always kept)

---

## Modeling

Per eligible PPG: chronological 80/20 split, fit log-log OLS, semi-log OLS
(if sign retry needed), LightGBM. Pick lowest-WAPE winner among sign-correct
candidates.

### Client questions

- **Q: Why is my biggest PPG skipped?**
  A: Modeling requires ≥20 observations after the chronological 80/20 split.
  Increase weeks of history.

- **Q: What models are tried?**
  A: Log-log OLS first. If the elasticity sign is wrong, a semi-log OLS is
  fit as a retry. LightGBM is always fit. Winner is the lowest test-WAPE
  among sign-correct candidates.

- **Q: Why was LightGBM picked as winner?**
  A: It produced the lowest test-set WAPE among models with the correct
  elasticity sign. LightGBM elasticities come from numerical derivatives at
  training points.

- **Q: What is the sign retry?**
  A: If log-log OLS gives a positive own-price coefficient (wrong sign for
  typical demand), we refit as a semi-log model. If both have wrong signs,
  LightGBM may still win and we surface `sign_ok=False`.

- **Q: Why is elasticity reported both positive and negative in different
  views?**
  A: Internally we store the signed coefficient (negative for
  downward-sloping demand). UI sometimes shows magnitude; check the field
  name.

- **Q: Why does the test set look so small?**
  A: Chronological 80/20 holdout. Short PPGs get 4–5 test weeks. Larger
  panels naturally get wider hold-outs.

- **Q: What's hierarchical shrinkage?**
  A: Partial pooling toward the category mean elasticity. Noisy PPG estimates
  shrink toward the cohort; clean ones move less. Reduces over-fitting.

- **Q: Why is my elasticity exactly -1.0?**
  A: Coincidence or a thin fit. Check the std_err — wide intervals point to
  a poorly identified coefficient.

### Corner cases

- **Fewer than 20 weeks per PPG** → PPG skipped; no contribution to
  downstream stages.
- **Wrong-sign log-log AND wrong-sign semi-log** → LightGBM may still win on
  WAPE; `sign_ok=False` is surfaced.
- **Only `log_price` as a feature** → minimum viable; fit still runs.
- **LightGBM std_err derived from numerical perturbation** → excluded from
  hierarchical pooling (not a sampling SE).
- **Perfectly collinear features (rare after refine)** → OLS produces NaN
  coefficients; LightGBM still fits.
- **All test rows have units = 0** → WAPE undefined; PPG fails downstream
  verdict.
- **NaN in any feature row** → row dropped before fitting.
- **LLM unreachable** → empty narrative; model fits and verdicts unchanged.

### Engineering notes

- Train/test split: chronological 80/20
- Minimum rows: 20 observations per PPG
- Sign-retry semi-log: `core/agents/modeling.py:98-121`
- Approval gate: `core/orchestrator/state.py`
- Per-agent model override: `MODEL_MODELING` env

---

## Results Reasoning

Run sanity checks on winners: sign correctness, elasticity magnitude band,
R² floor, WAPE ceiling. Verdict-aggregate per PPG.

### Client questions

- **Q: What makes a PPG 'pass'?**
  A: All four checks clear: `sign_ok=True`, magnitude in [0.3, 6.0], R² ≥
  0.30, WAPE ≤ 0.30.

- **Q: Why is my PPG 'warn'?**
  A: One or more thresholds breached but sign is correct. Magnitude outside
  band, R² below 0.30, or WAPE > 0.30 but < some fatal level.

- **Q: Why is 'fail' rare?**
  A: It triggers when sign is wrong or the PPG was skipped at modeling. Most
  live PPGs land in pass or warn.

- **Q: What is the magnitude band [0.3, 6.0]?**
  A: Typical CPG own-price elasticity range from published literature.
  Outside this band, results are suspicious and surfaced as warn.

- **Q: R² ≥ 0.30 seems low. Why?**
  A: Panel-data noise is high (store-level seasonality, week-to-week
  swings). 0.30 means the model explains roughly a third of variance — a
  respectable floor.

- **Q: What if R² is 0.95?**
  A: Passes here, but check the holdout WAPE in validation — over-fitting
  will inflate in-sample R².

- **Q: Does this stage refit anything?**
  A: No. It consumes modeling artifacts and produces verdicts only.

- **Q: How do verdicts feed downstream?**
  A: Surfaced in the insights summary. They do not gate optimization, but
  they colour the recommendation confidence.

### Corner cases

- **PPG skipped at modeling** → automatic fail at results_reasoning.
- **WAPE > 0.30 but sign and magnitude are fine** → verdict: warn (not fail).
- **Sign wrong but magnitude in band** → verdict: fail. Sign-correctness
  dominates.
- **R² = 0 (no variance explained)** → verdict: fail.
- **Magnitude exactly 0.30** → passes (inclusive lower bound).
- **LightGBM winner with no signed elasticity** → surfaced as 'no signed
  estimate'; sign check skipped, others apply.
- **All PPGs fail** → insights shows a red banner; no recommendations
  produced.
- **LLM unreachable** → verdicts deterministic; narrative blank.

### Engineering notes

- Thresholds: `core/config.py` `ValidationThresholds`
- Override knobs: `VALIDATION__SIGN_PASS`, `VALIDATION__WAPE_PASS` env vars
- Verdict logic: worst-status precedence (fail > warn > pass)

---

## Decomposition

Refit winners on the full series; decompose observed units into base + driver
groups (price, promo, distribution, seasonality, …) + residual.

### Client questions

- **Q: Why don't the components sum exactly to observed units?**
  A: Residual term absorbs un-modelled variance. Reconciliation tolerance is
  reported; large mismatches drop confidence.

- **Q: What is 'base'?**
  A: Predicted units when all drivers sit at reference values: no promo,
  mean ACV, mean competitor price, etc.

- **Q: Why is the price contribution sometimes positive?**
  A: Price below the reference creates a lift (positive contribution). Price
  above creates drag (negative).

- **Q: Promo contribution for non-promo weeks is zero — is that right?**
  A: Yes. We use group-wise ablation: turn off promo features and see the
  unit delta. Off-weeks already have zero, so no delta.

- **Q: Is decomposition per-PPG or panel?**
  A: Per-PPG. Each winner is refit on the full series (no holdout) and
  decomposed at PPG×week grain.

- **Q: LightGBM decomposition feels coarse compared to OLS — why?**
  A: LightGBM uses group-wise ablation, not closed-form coefficients. No
  per-feature granularity; only per-group totals.

- **Q: Why is seasonality flat?**
  A: Only fires if week-of-year features survived refine. If they were
  pruned, the seasonality group is absent.

- **Q: Can I trust the absolute magnitudes?**
  A: Within a PPG, yes — proportions are consistent. Across PPGs the
  absolute scale tracks the model's R².

### Corner cases

- **Reconciliation error > 1%** → confidence drops toward zero; surfaced in
  the audit table.
- **Non-OLS / non-LightGBM model** → skipped at decomposition; no
  contribution rows produced.
- **Single-week PPG (degenerate)** → trivial decomposition: only base, no
  driver groups.
- **Negative base value** → model artifact; surfaced as an anomaly in the
  table.
- **All weeks were promo** → base inherits the promo state; price
  contribution diluted.
- **Missing driver group** (e.g. no competitor price) → group entry is 0,
  not NaN. UI shows 0 contribution.
- **PPG skipped at modeling** → not decomposed.
- **LLM unreachable** → headline narrative falls back to template.

### Engineering notes

- OLS path (closed-form): `core/agents/decomposition.py`
- LightGBM path (ablation): group-wise feature holdout
- Reconciliation tolerance: 1% target

---

## Simulation

Sweep price × promo grid per PPG. OLS path uses closed-form prediction;
LightGBM path refits and runs the predictor over the grid. Produces warm-start
for optimization.

### Client questions

- **Q: What price grid is swept?**
  A: 13 multipliers (e.g. 0.85 to 1.15 in steps) × 2 promo states (on/off) =
  26 cells per PPG.

- **Q: Why are out-of-range predictions flat?**
  A: LightGBM clamps at the training-price envelope. Extrapolating outside
  observed prices would be unreliable.

- **Q: Can I sweep more than promo on/off?**
  A: Not yet. Depth, duration, and lift modelling are on the roadmap.

- **Q: What is the 'base price' here?**
  A: Median observed price per PPG. LightGBM also reads a `log_base_price`
  feature where available.

- **Q: Does simulation respect competitor prices?**
  A: Competitor price is held at the mean during the sweep. We don't co-vary
  it.

- **Q: Why does the heatmap look weird at extremes?**
  A: Predictions flatten at the price envelope edges. The optimizer clips
  those rungs from the ladder.

- **Q: Does simulation use train or full data?**
  A: Full data, same refit as decomposition. No holdout.

- **Q: Can I export the grid?**
  A: Yes — `simulation_grid.json` is downloadable from the artifact list.

### Corner cases

- **PPG skipped at modeling** → no simulation grid produced.
- **All features held at mean for LightGBM sweep** → interaction effects may
  look unrealistic; UI flags low confidence.
- **Multiplier × base price below cost** → cell still simulated; optimization
  will floor it on margin.
- **Promo=on but PPG never had promo history** → prediction is extrapolated
  blindly; surface as low-confidence cell.
- **Competitor price column entirely null** → competitor coefficient
  dropped; sweep proceeds without it.
- **Very narrow observed price range** → grid resolution stays at 13
  multipliers; cells outside envelope are clipped at optimization.
- **NaN in context features** → forward-filled or replaced with mean before
  sweep.
- **`simulation_summary` smaller than `simulation_grid`** → expected. Summary
  is the highlights table; grid is the full sweep.

### Engineering notes

- Grid shape: 13 multipliers × 2 promo states = 26 cells per PPG
- Envelope clipping: `core/agents/optimization.py:169-247` (applied at
  optimization stage)
- OLS path (closed-form): `core/agents/simulation.py`

---

## Optimization

Solve continuous warm-start, then MILP on discrete price ladder under margin
floor, competitor gap, move guardrail. Soft-relax if strict MILP is
infeasible.

### Client questions

- **Q: Why does my recommended price differ from the continuous optimum?**
  A: We enforce a discrete price ladder. The MILP picks the best ladder rung,
  which may step away from the continuous optimum.

- **Q: What is the margin floor?**
  A: Default: 5% above cost. Configurable per run via the constraint editor
  in the approval gate.

- **Q: Can I customize the price ladder?**
  A: Yes. Edit ladder values in the optimization approval gate and rerun the
  stage. The orchestrator re-solves without re-modelling.

- **Q: What does 'relaxed' mean?**
  A: Strict MILP was infeasible. We re-solve with soft constraints, surface
  which constraints bound, and report the trade-off.

- **Q: How do I find which constraints conflicted?**
  A: Check `optimization_constraints.json` artifact — the `binding_violations`
  array lists each.

- **Q: Why didn't price go to the cheapest ladder rung?**
  A: Likely the margin floor or competitive gap blocked it. Inspect those
  constraints.

- **Q: What's 'envelope clipped'?**
  A: LightGBM-only safeguard: ladder rungs outside the training price range
  are dropped before solving. Prevents extrapolation.

- **Q: Can I optimize for revenue instead of margin?**
  A: Yes, set `options.objective='revenue'` at rerun time. Default is margin.

### Corner cases

- **Margin floor exceeds the only ladder rung satisfying comp gap** →
  infeasible MILP; soft-relaxation fires; binding violations surfaced.
- **Cost data missing** → fallback: 55% of base price assumed as cost.
- **Single-rung ladder** → trivially solved.
- **LightGBM with all rungs outside training envelope** → keep full ladder;
  expect low confidence and rely on validation verdict.
- **PPG skipped at modeling** → not optimized; no recommendation produced.
- **Move guardrail conflicts with margin floor** → relax fires; binding
  violation reported.
- **Identical ladders across PPGs** → same constraints applied; OK — no
  special handling.
- **Rerun without changing options** → idempotent — same result.

### Engineering notes

- MILP solver: `core/optimization/milp.py:130-242` (PuLP)
- Envelope clipping: `core/agents/optimization.py:169-247`
- Rerun support: `POST /runs/{id}/rerun` with `options.optimization`
- Default cost: 55% of base price
- Approval gate: `core/orchestrator/state.py` `APPROVAL_GATES`

---

## Validation

Rolling-origin cross-validation. Fit winner family per fold; check sign
stability, mean WAPE, elasticity CV, magnitude band.

### Client questions

- **Q: What is rolling-origin cross-validation?**
  A: Train set expands; test set slides forward in time. Four folds by
  default. Mimics how the model will be used in production.

- **Q: Why does my sign stability score change on rerun?**
  A: It shouldn't — folds are deterministic given the data. If your data
  changed (new weeks added), folds shift.

- **Q: What is elasticity CV?**
  A: Coefficient of variation of the elasticity estimate across folds. CV ≤
  0.40 indicates a stable estimate.

- **Q: Why is mean WAPE different from the modeling-stage WAPE?**
  A: Modeling uses a single 80/20 split. Validation uses four sliding folds;
  the average is over multiple holdouts.

- **Q: Can I trust a 'warn' PPG for pricing decisions?**
  A: With caveats. Inspect which check warned — wide elasticity CV is more
  concerning than a small magnitude excursion.

- **Q: What does 'magnitude check fails but sign and WAPE pass' mean?**
  A: Mean absolute elasticity lies outside [0.3, 6.0]. Either too inelastic
  or too elastic for typical CPG.

- **Q: Why are residuals shown?**
  A: Diagnostic. Look for non-random patterns (trends, fan-out) that suggest
  mis-specification.

- **Q: Does validation refit the winner model?**
  A: Yes — same family, refit per fold. Hyperparameters carry over.

### Corner cases

- **Fewer than 4 fold-points available** → fewer folds used; `n_folds`
  reported.
- **Single fold pass** → sign stability check is degenerate (only 1 fold).
- **All folds fail** → verdict: fail; PPG flagged unusable.
- **Wrong sign in only one fold** → sign stability may still pass if ≥75%
  folds correct.
- **LightGBM in validation** → no closed-form signed elasticity; sign
  stability derived via local decomposition.
- **PPG skipped at modeling** → not validated.
- **Residuals identically zero** → over-fit signal; surfaces but does not
  fail directly.
- **LLM unreachable** → numerical verdicts unchanged; narrative blank.

### Engineering notes

- Rolling-origin CV: `core/agents/validation.py`
- Default folds: 4 (configurable)
- Thresholds (shared with results_reasoning): `core/config.py`
  `ValidationThresholds`

---

## Insights

Read-only assembly of per-PPG recommendations, verdicts, and the executive
narrative. Renders HTML and PDF report.

### Client questions

- **Q: Where do the numbers in the report come from?**
  A: Every figure traces back to an upstream artifact. Insights does not
  refit or recompute — only assembles.

- **Q: Why is the PDF missing?**
  A: WeasyPrint failed to render (missing system fonts, library not
  installed, etc.). HTML is still available.

- **Q: What is 'total revenue uplift'?**
  A: Sum of revenue delta across optimized PPGs, comparing recommended prices
  to baseline prices.

- **Q: Can I filter the report by category or brand?**
  A: Not yet via UI. Filter the underlying artifacts post-run.

- **Q: Why are some PPGs missing from the report?**
  A: Skipped upstream — either ineligible at PPG selection or unmodelled due
  to thin data.

- **Q: What does the overall confidence score mean?**
  A: `n_pass / n_validated`. Drops to zero if validation didn't run for any
  PPG.

- **Q: Why does the headline narrative use 'approximately'?**
  A: When the LLM is in dry-run mode (no API key), narrative comes from a
  deterministic template that hedges intentionally.

- **Q: Where's the cost dashboard?**
  A: Separate artifact `cost_summary.json`. Rendered in the run sidebar; not
  embedded in the PDF.

### Corner cases

- **All PPGs failed verdict** → red banner; no recommendations section.
- **WeasyPrint missing on host** → PDF artifact absent; HTML still served.
- **Validation stage skipped entirely** → confidence score falls back to 0.
- **No optimization results** → recommendations section blank; headline
  still shows.
- **Run rerun after editing constraints** → numbers refresh; report
  regenerated.
- **Live card numbers differ from report** → cards are live; report is a
  snapshot at insights-run time.
- **Large run (>20 PPGs)** → tables paginate in HTML; PDF flows naturally
  across pages.
- **LLM unreachable** → headline uses deterministic template; metrics
  unchanged.

### Engineering notes

- Assembly only: no refit — reads upstream artifacts
- PDF renderer: WeasyPrint (optional dep)
- Confidence formula: `n_pass / n_validated`
- Cost summary: `cost_summary.json`
