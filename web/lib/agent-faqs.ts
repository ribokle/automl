import type { AgentName } from "./types";

export interface StageFAQ {
  q: string;
  a: string;
}

export interface StageCornerCase {
  condition: string;
  behaviour: string;
  userSees?: string;
}

export interface EngineeringNote {
  topic: string;
  ref: string;
}

export interface StageFAQEntry {
  questions: StageFAQ[];
  cornerCases: StageCornerCase[];
  engineeringNotes: EngineeringNote[];
  docsAnchor: string;
}

export const STAGE_FAQS: Record<AgentName, StageFAQEntry> = {
  ingestion: {
    docsAnchor: "ingestion",
    questions: [
      {
        q: "Why did the run fail before any models were fit?",
        a: "Ingestion enforces minimums: ≥500 rows, ≥5 unique SKUs, ≥1 store. Below any floor the run halts. Widen the panel (more weeks or SKUs) and retry.",
      },
      {
        q: "My price column failed validation but the numbers look fine.",
        a: "Price mean must fall in $0.50–$50.00 and stdev in $0.00–$20.00. Common cause: prices in cents rather than dollars, or a single SKU with an extreme outlier price.",
      },
      {
        q: "Why are my units flagged as outliers when they're real?",
        a: "Q50 caps at 10k units, Q95 at 50k. Hypermarket-scale SKUs trip the warn. Either confirm and ignore, or split the file by channel.",
      },
      {
        q: "What does the drift warning mean?",
        a: "Each numeric column's mean and Q25/Q75 are compared against a baseline snapshot. Drift outside ±40% triggers a warn. First runs have no baseline so drift is skipped.",
      },
      {
        q: "Why is base_price flagged when it's lower than transaction price?",
        a: "Base price is the regular shelf price; transaction price is what shoppers paid (incl. promo). Base must be ≥ transaction in ≥99% of weeks. Reversed columns trip this.",
      },
      {
        q: "Can I run without distribution_acv?",
        a: "Yes — the column is optional. If present it must be 0–100. If missing, the distribution coefficient simply drops out of the models.",
      },
      {
        q: "What if I have negative units (returns)?",
        a: "Hard fail. Aggregate net units at the same grain or remove return rows upstream — the panel mart assumes non-negative weekly units.",
      },
      {
        q: "Why did the LLM narrative differ across two runs of the same file?",
        a: "Only the narrative is LLM-generated. Thresholds and verdicts are deterministic. With no API key (or on non-JSON output) the narrative falls back to a template that is identical across runs.",
      },
    ],
    cornerCases: [
      {
        condition: "All weeks promo (TPR=1 everywhere)",
        behaviour: "Ingestion passes; flagged downstream at PPG eligibility, not here.",
        userSees: "Eligibility score low at PPG selection.",
      },
      {
        condition: "Single-store dataset",
        behaviour: "Ingestion passes (≥1 store satisfied); ACV becomes a constant.",
        userSees: "Distribution coefficient identified as singular at modeling.",
      },
      {
        condition: "Missing weeks (date gaps)",
        behaviour: "Ingestion passes; EDA surfaces the coverage shortfall.",
        userSees: "Coverage component pulls PPG eligibility score down.",
      },
      {
        condition: "Mixed currencies in one file",
        behaviour: "Not detected directly; price-distribution warn may fire if range straddles $0.50–$50.",
        userSees: "Wide price-stdev warning.",
      },
      {
        condition: "Columns in mixed case (SKU vs sku)",
        behaviour: "dbt source normalises to lower-case; both resolve.",
      },
      {
        condition: "Zero-variance price column",
        behaviour: "Ingestion passes; price_cv=0 marks every PPG ineligible at stage 3.",
      },
      {
        condition: "Upload exceeds MAX_UPLOAD_MB (default 200)",
        behaviour: "API rejects upload with HTTP 413 before ingestion runs.",
        userSees: "Upload error in UI; raise MAX_UPLOAD_MB env to allow.",
      },
      {
        condition: "Non-UTF8 CSV (e.g. Windows-1252)",
        behaviour: "DuckDB read fails before checks run; surfaced as ingestion error.",
      },
    ],
    engineeringNotes: [
      { topic: "Threshold definitions", ref: "core/data/expectations.py:18-107" },
      { topic: "Drift slack (DRIFT_SLACK_PCT env)", ref: "core/config.py drift_slack_pct" },
      { topic: "Baseline path (BASELINE_DIR env)", ref: "core/config.py baseline_dir" },
      { topic: "LLM dry-run fallback", ref: "core/agents/ingestion.py _dry_run_*" },
      { topic: "Required columns", ref: "sku, store_id, week_start, units, price, tpr_flag" },
    ],
  },

  ppg_mapping: {
    docsAnchor: "ppg-mapping",
    questions: [
      {
        q: "Why are these two SKUs in different PPGs when they're the same product?",
        a: "Brand and category form a hard partition — different brand or category SKUs never merge. Check those columns for spelling or case mismatches.",
      },
      {
        q: "Why didn't the premium tier get its own PPG?",
        a: "Splitting needs at least 4 SKUs in the brand-category and a price spread of ≥0.50 log-dollars (~65% step), plus silhouette score ≥0.70. Narrow tiers stay merged.",
      },
      {
        q: "What does the confidence score on a SKU mean?",
        a: "1 − (distance-to-own-centroid / distance-to-nearest-other-centroid), clipped to [0,1]. Single-cluster default is 0.92.",
      },
      {
        q: "Can I edit the PPGs manually?",
        a: "Not directly today — but the approval gate after this stage lets you reject and rerun ingestion with cleaned inputs.",
      },
      {
        q: "Why is one PPG flagged but still eligible?",
        a: "A flag means within-group price ratio exceeds 2×. The cluster is internally heterogeneous; we keep it but surface for review.",
      },
      {
        q: "The LLM rationale changed between runs. Is the mapping unstable?",
        a: "No. Clustering is fully deterministic (k-means with fixed seed). Only the narrative text varies if the LLM is in API mode.",
      },
      {
        q: "Should I rerun if I add new SKUs?",
        a: "Yes — full reclustering happens, and PPG IDs may shift. Save prior recommendations before rerunning.",
      },
      {
        q: "Does seasonality affect PPG mapping?",
        a: "No. PPGs are static across the whole period; seasonality enters at the modeling stage as week-of-year features.",
      },
    ],
    cornerCases: [
      {
        condition: "Single SKU in a brand/category",
        behaviour: "Single-SKU PPG with confidence 0.92.",
      },
      {
        condition: "2–3 SKUs in a brand/category",
        behaviour: "Single PPG (below the 4-SKU k-means floor).",
      },
      {
        condition: "All SKUs at identical price",
        behaviour: "Single PPG (log-price spread <0.50 below split threshold).",
      },
      {
        condition: "Silhouette score below 0.70 after attempted split",
        behaviour: "Split rejected; cluster kept whole.",
      },
      {
        condition: "New SKU added between runs",
        behaviour: "Full re-clustering; prior PPG IDs may shift.",
        userSees: "Recommendation table is keyed by new PPG IDs.",
      },
      {
        condition: "Same pack size, different brands",
        behaviour: "Never merged — brand partition is hard.",
      },
      {
        condition: "Missing brand or category column",
        behaviour: "Ingestion catches it first; never reaches PPG mapping.",
      },
      {
        condition: "Within-group price ratio >2×",
        behaviour: "PPG kept but `flagged=true`; surfaced in eligibility narrative.",
      },
    ],
    engineeringNotes: [
      { topic: "Clustering rules", ref: "core/ppg/cluster.py:33-98" },
      { topic: "Confidence formula", ref: "core/ppg/cluster.py" },
      { topic: "LLM dry-run rationales", ref: "core/agents/ppg_mapping.py _dry_run_*" },
      { topic: "Approval gate", ref: "core/orchestrator/state.py APPROVAL_GATES" },
    ],
  },

  ppg_selection: {
    docsAnchor: "ppg-selection",
    questions: [
      {
        q: "Why is my best-selling PPG marked ineligible?",
        a: "Composite score below 0.60. Inspect components: volume, coverage, price CV, and promo %. Any one dragging hard can sink the total.",
      },
      {
        q: "What threshold makes a PPG eligible?",
        a: "Overall score ≥ 0.60. Weights: volume 25%, coverage 25%, price CV 30%, promo % 20%. Threshold is 0.60 inclusive.",
      },
      {
        q: "My PPG has only 60% week coverage — is that enough?",
        a: "Coverage component scores below 0.60 there, but other components can offset. Always check the composite score, not a single component.",
      },
      {
        q: "We never promote. Can I still model?",
        a: "Promo % expected in [5%, 50%]. Below 5% lowers the promo component to near zero. Below ~3% the PPG usually falls under the 0.60 threshold.",
      },
      {
        q: "Why is a PPG with massive volume scored low?",
        a: "Volume saturates at 50k units. Beyond that you get full 0.25 credit but nothing extra — other components must still clear their bands.",
      },
      {
        q: "Can I lower the eligibility threshold?",
        a: "Not via UI yet. Edit `core/ppg/score.py` defaults or override in code. Future enhancement: per-run override via approval gate.",
      },
      {
        q: "What is price CV?",
        a: "Coefficient of variation (stdev / mean) of price across observed weeks. We saturate at 0.30; below 0.12 the score drops sharply.",
      },
      {
        q: "Do eligibility scores update when I edit constraints?",
        a: "No. Eligibility is tied to the input data, not the optimization constraints. Editing constraints only re-solves the optimization stage.",
      },
    ],
    cornerCases: [
      {
        condition: "Zero price variation (price_cv=0)",
        behaviour: "Price-CV component pulls overall score to ~0.40; PPG marked ineligible.",
      },
      {
        condition: "100% promo weeks (TPR always on)",
        behaviour: "Promo % outside [5%,50%] band; score drops; usually ineligible.",
      },
      {
        condition: "Fewer than 5 weeks of data",
        behaviour: "Coverage near zero; ineligible.",
      },
      {
        condition: "Volume in millions of units",
        behaviour: "Saturates at 50k units (full credit); no penalty.",
      },
      {
        condition: "Very high price CV (>0.5)",
        behaviour: "Saturates at 0.30 (full credit); not penalised for being high.",
      },
      {
        condition: "All PPGs ineligible",
        behaviour: "Modeling skips every PPG; downstream stages cascade with empty results.",
        userSees: "Insights shows a red banner — no eligible PPGs.",
      },
      {
        condition: "LLM unreachable",
        behaviour: "Per-PPG rationales fall back to deterministic templates.",
      },
      {
        condition: "Score exactly 0.60",
        behaviour: "Eligible (inclusive lower bound).",
      },
    ],
    engineeringNotes: [
      { topic: "Scoring formula", ref: "core/ppg/score.py:27-106" },
      { topic: "Threshold (0.60)", ref: "core/ppg/score.py ELIGIBILITY_THRESHOLD" },
      { topic: "Component weights", ref: "vol 25, cov 25, price_cv 30, promo 20" },
    ],
  },

  feature_selection: {
    docsAnchor: "feature-selection",
    questions: [
      {
        q: "Why was log_units picked as the target?",
        a: "Heuristic match on canonical column names (`units`, `sales`, `qty`) plus numeric type. We then log-transform inside engineering.",
      },
      {
        q: "Which columns are classified as identifier?",
        a: "`sku`, `store_id`, `ppg_id`, and any column with an `_id` suffix or constant within row.",
      },
      {
        q: "Why is `week_start` flagged as the temporal column?",
        a: "Datetime-parseable plus name match. If parsing fails, the column drops back to numeric/categorical.",
      },
      {
        q: "Are categorical columns kept?",
        a: "Yes, passed through to feature engineering. Low-cardinality categoricals get dummy-encoded; high-cardinality ones are dropped.",
      },
      {
        q: "Can I override the classification?",
        a: "Not via UI today. Pre-clean your file or extend `core/agents/feature_selection.py`.",
      },
      {
        q: "Why is a 0/1 column treated as numeric?",
        a: "Only columns suffixed `_flag` or with exactly 2 distinct values 0/1 are flagged. Other binary integers are kept numeric.",
      },
      {
        q: "What if my target is dollars not units?",
        a: "Pipeline still runs; elasticity will then be a revenue elasticity (often closer to 1.0 in magnitude). Interpret accordingly.",
      },
      {
        q: "Why so few columns selected as numeric?",
        a: "High-cardinality strings and date columns are filtered out. Recheck for stringified numbers (e.g. price stored as text).",
      },
    ],
    cornerCases: [
      {
        condition: "No clear target column",
        behaviour: "Stage fails with `target column not detected`; rename your sales/units column.",
      },
      {
        condition: "Two candidates for target (e.g. both units and sales)",
        behaviour: "First by configured priority wins; the other becomes a numeric feature.",
      },
      {
        condition: "Date column with non-ISO format",
        behaviour: "May parse as identifier; downstream EDA breaks.",
        userSees: "EDA fails with `temporal column unparseable`.",
      },
      {
        condition: "Boolean columns stored as 'true'/'false' strings",
        behaviour: "Not detected as flag; treated as categorical.",
      },
      {
        condition: "Numeric column with <2 unique values",
        behaviour: "Classified as flag if values are 0/1, otherwise dropped.",
      },
      {
        condition: "All-null columns",
        behaviour: "Excluded silently; not reported as candidates.",
      },
      {
        condition: "Mixed dtype within a column",
        behaviour: "Pandas coerces to object; usually misclassified as categorical.",
      },
      {
        condition: "LLM unreachable",
        behaviour: "Deterministic template narrative; classification unchanged.",
      },
    ],
    engineeringNotes: [
      { topic: "Classification logic", ref: "core/agents/feature_selection.py" },
      { topic: "Heuristic priorities", ref: "target → flag → numeric → categorical" },
    ],
  },

  eda: {
    docsAnchor: "eda",
    questions: [
      {
        q: "Why is the Spearman correlation NaN for some features?",
        a: "Spearman requires at least 30 non-null paired observations. Sparse columns drop out as NaN.",
      },
      {
        q: "What threshold marks a relationship as 'strong'?",
        a: "Display only: |ρ| ≥ 0.6 strong, ≥ 0.3 moderate. These are visual cues, not gates.",
      },
      {
        q: "Why is missingness high for lag_units_4?",
        a: "The first 4 weeks of each PPG's series produce NaN lags by construction. Expected.",
      },
      {
        q: "Are correlations computed within PPG or across the panel?",
        a: "PPG × week panel, pooled across PPGs. Within-PPG views are available in downstream visuals.",
      },
      {
        q: "Pearson vs Spearman — which should I trust?",
        a: "Spearman is primary because elasticity relationships are non-linear in raw units. Pearson is reported for reference.",
      },
      {
        q: "What does 'candidates ranked' mean?",
        a: "Features sorted by absolute Spearman ρ against the target. Top of the list = strongest monotonic signal.",
      },
      {
        q: "Why isn't price in the top correlates?",
        a: "Price-unit relationship is most cleanly seen in log-log space. The raw correlation can be modest while the log elasticity is sharp.",
      },
      {
        q: "Why does EDA show fewer features than engineering produces?",
        a: "EDA ranks candidates only. Engineering keeps the full feature set; pruning happens later in refine.",
      },
    ],
    cornerCases: [
      {
        condition: "Single PPG",
        behaviour: "All cross-PPG variance collapses; correlations inflate.",
        userSees: "Banner: 'single-PPG run — interpret correlations with care'.",
      },
      {
        condition: "Zero-variance numeric column",
        behaviour: "Excluded from correlation table.",
      },
      {
        condition: "All-null column",
        behaviour: "Excluded; not surfaced.",
      },
      {
        condition: "n = 29 paired observations",
        behaviour: "Spearman returns NaN (one below the 30-row floor).",
      },
      {
        condition: "Categorical with high cardinality",
        behaviour: "Sampled to top-k levels; full distribution not shown.",
      },
      {
        condition: "Mostly-zero feature (e.g. competitor_price 90% null)",
        behaviour: "Low n after dropping nulls; usually NaN ρ.",
      },
      {
        condition: "Multi-collinear cluster of features",
        behaviour: "All show similar high ρ; pruning lands in feature_refine.",
      },
      {
        condition: "LLM unreachable",
        behaviour: "Insights paragraph empty; tables still render.",
      },
    ],
    engineeringNotes: [
      { topic: "Spearman implementation", ref: "core/agents/eda.py" },
      { topic: "Minimum n threshold", ref: "n ≥ 30 for ρ" },
      { topic: "Display thresholds (UI only)", ref: "web/components/tables/TargetRelationship.tsx" },
    ],
  },

  advanced_eda: {
    docsAnchor: "advanced-eda",
    questions: [
      {
        q: "Why are only the top-K PPGs analysed?",
        a: "STL, ACF, isolation-forest and PELT all scale per-series; on Dominick's-sized data running them on every PPG is wasteful. The agent caps heavy analysis at the top-K by revenue (default 50, configurable via run.options['advanced_eda']['max_series']) and emits cheap summaries for the rest.",
      },
      {
        q: "Is the price-volume slope the same as elasticity?",
        a: "No. It's an uncontrolled univariate log-log slope on (log price, log units). The modelling agent recovers the controlled estimate with the full feature set. Use the slope as a sign-check, not a number.",
      },
      {
        q: "Why is the cross-PPG correlation labelled 'demand correlation' not 'cannibalisation'?",
        a: "At EDA stage we can't separate substitution from common-cause drivers (holiday, weather, supply). The matrix is a category-coherence check; cannibalisation evidence requires the controlled cross-elasticities the decomposition agent produces.",
      },
      {
        q: "How is a stockout detected?",
        a: "units = 0 with distribution_acv > 0 AND the prior week had positive units at unchanged price. Pure heuristic — picks up classic out-of-stock weeks without flagging delisting events.",
      },
      {
        q: "What's a change point?",
        a: "PELT detects step changes in baseline price per PPG. The modelling agent should treat weeks across a change point as different regimes — fitting a single elasticity across a relaunch is dangerous.",
      },
    ],
    cornerCases: [
      {
        condition: "Series shorter than 2 × period",
        behaviour: "STL is skipped for that PPG; ACF/PACF and stationarity still run if n ≥ 20.",
        userSees: "STL panel shows 'insufficient data'.",
      },
      {
        condition: "Single-store panel",
        behaviour: "store_variability sidecar is empty; everything else is unaffected.",
      },
      {
        condition: "No promo columns",
        behaviour: "Promo lift / promo calendar artefacts are written but empty; downstream UI sections collapse gracefully.",
      },
      {
        condition: "No holiday column",
        behaviour: "holiday_lift.json contains an empty rows array.",
      },
      {
        condition: "LLM unreachable",
        behaviour: "Deterministic dry-run summary fills the findings + narrative; artefact contents are unchanged.",
      },
    ],
    engineeringNotes: [
      { topic: "Pure stats helpers", ref: "core/features/advanced_eda.py" },
      { topic: "Agent + artefact orchestration", ref: "core/agents/advanced_eda.py" },
      { topic: "Frontend dashboard", ref: "web/app/runs/[id]/eda + web/components/AdvancedEDADashboard.tsx" },
      { topic: "Compute caps", ref: "run.options['advanced_eda'] = {max_series, corr_cap}" },
    ],
  },

  feature_engineering: {
    docsAnchor: "feature-engineering",
    questions: [
      {
        q: "What features get built?",
        a: "log_price, log_units, promo and price lags at lag-1 and lag-4, week-of-year seasonality dummies, and distribution_acv passthrough.",
      },
      {
        q: "Why lag 1 and lag 4 specifically?",
        a: "Lag 1 captures last-week carryover; lag 4 covers a 4-week consumption cycle common in CPG. Hard-coded today; configurable later.",
      },
      {
        q: "Why is parquet sometimes CSV?",
        a: "pyarrow is optional. If unavailable, we transparently fall back to CSV. Downstream readers handle either format.",
      },
      {
        q: "Are competitor prices included?",
        a: "Yes if the column is present in your CSV. The join is left-outer and nullable; missing competitor data leaves the feature null.",
      },
      {
        q: "How are missing lag values handled?",
        a: "First weeks per PPG produce NaN lags. The modeling stage drops those rows before fitting.",
      },
      {
        q: "Are holidays included?",
        a: "Implicitly via week-of-year dummies. Explicit holiday flags (Thanksgiving, Easter) are on the roadmap.",
      },
      {
        q: "Can I add custom features?",
        a: "Not via UI. Subclass `FeatureEngineering` or fork the agent.",
      },
      {
        q: "Why is base_price log-transformed but distribution_acv linear?",
        a: "Prices follow log-log elasticity; shares are bounded [0,1] and naturally linear. This is the modelling convention.",
      },
    ],
    cornerCases: [
      {
        condition: "Series shorter than max lag (4 weeks)",
        behaviour: "All lag values NaN; modeling stage skips that PPG.",
      },
      {
        condition: "Promo never active for a PPG",
        behaviour: "Promo lags all zero; coefficient unidentified during fit.",
      },
      {
        condition: "log(0) units would be undefined",
        behaviour: "Units below 1 are floored to 1 before the log transform.",
      },
      {
        condition: "Parquet write fails (pyarrow missing)",
        behaviour: "CSV fallback; downstream readers handle either path.",
      },
      {
        condition: "Single-store ACV (constant)",
        behaviour: "Coefficient unidentified at fit time; reported as singular.",
      },
      {
        condition: "distribution_acv missing entirely",
        behaviour: "Column dropped; no distribution coefficient produced.",
      },
      {
        condition: "Two-week file",
        behaviour: "Almost everything NaN; modeling skips most PPGs.",
      },
      {
        condition: "Year boundary crossing in week-of-year",
        behaviour: "Handled with modular (cyclic) dummies; no off-by-one.",
      },
    ],
    engineeringNotes: [
      { topic: "Feature list", ref: "core/agents/feature_engineering.py ENGINEERED_COLUMNS" },
      { topic: "Lag values", ref: "lag 1, lag 4 (hard-coded)" },
      { topic: "Parquet fallback", ref: "auto-detects pyarrow; CSV otherwise" },
    ],
  },

  feature_refine: {
    docsAnchor: "feature-refine",
    questions: [
      {
        q: "Why was my favourite feature dropped?",
        a: "Either VIF > 10 (multicollinear with the rest) or |corr| > 0.95 with a peer. Both indicate redundancy.",
      },
      {
        q: "What is VIF?",
        a: "Variance Inflation Factor: how much a feature's variance is inflated by collinearity with other features. >10 is the conventional pruning threshold.",
      },
      {
        q: "Is log_price always kept?",
        a: "Yes. log_price is protected — it's the elasticity predictor and never gets dropped, even if its VIF is high.",
      },
      {
        q: "Why are the kept and dropped lists empty?",
        a: "No engineered features survived earlier stages. Modeling will likely run on log_price only or skip.",
      },
      {
        q: "What's the |corr| threshold?",
        a: "0.95 absolute Pearson correlation. Pairs above this threshold have one member dropped (lower-VIF survives).",
      },
      {
        q: "Does refining happen per-PPG or pooled?",
        a: "Pooled across the whole panel. Every modeled PPG sees the same refined feature set.",
      },
      {
        q: "What's the difference between feature_selection and feature_refine?",
        a: "Selection picks raw columns to consider. Refine prunes the engineered set after lags/interactions are built. Two different stages.",
      },
      {
        q: "Can I override the kept set?",
        a: "Not via UI today. Edit `core/agents/feature_refine.py` to add protected features.",
      },
    ],
    cornerCases: [
      {
        condition: "Fewer than 2 features survive refine",
        behaviour: "Modeling falls back to log-log on price alone.",
      },
      {
        condition: "log_price has high VIF on its own",
        behaviour: "Still kept (protected); the VIF warning surfaces but does not drop it.",
      },
      {
        condition: "All lag features perfectly correlated",
        behaviour: "Most dropped; one representative survives.",
      },
      {
        condition: "Singular matrix during VIF computation",
        behaviour: "Feature flagged suspect and dropped.",
      },
      {
        condition: "Promo features near-zero variance",
        behaviour: "Dropped silently as non-informative.",
      },
      {
        condition: "Single PPG in run",
        behaviour: "Any PPG-level dummies trivially dropped.",
      },
      {
        condition: "Engineered frame missing target column",
        behaviour: "Stage fails with explicit error; do not proceed to modeling.",
      },
      {
        condition: "LLM narrative differs across runs",
        behaviour: "Expected — the kept/dropped lists themselves are deterministic.",
      },
    ],
    engineeringNotes: [
      { topic: "VIF threshold", ref: "VIF < 10" },
      { topic: "Correlation threshold", ref: "|corr| ≤ 0.95" },
      { topic: "Protected features", ref: "log_price (always kept)" },
    ],
  },

  modeling: {
    docsAnchor: "modeling",
    questions: [
      {
        q: "Why is my biggest PPG skipped?",
        a: "Modeling requires ≥20 observations after the chronological 80/20 split. Increase weeks of history.",
      },
      {
        q: "What models are tried?",
        a: "Log-log OLS first. If the elasticity sign is wrong, a semi-log OLS is fit as a retry. LightGBM is always fit. Winner is the lowest test-WAPE among sign-correct candidates.",
      },
      {
        q: "Why was LightGBM picked as winner?",
        a: "It produced the lowest test-set WAPE among models with the correct elasticity sign. LightGBM elasticities come from numerical derivatives at training points.",
      },
      {
        q: "What is the sign retry?",
        a: "If log-log OLS gives a positive own-price coefficient (wrong sign for typical demand), we refit as a semi-log model. If both have wrong signs, LightGBM may still win and we surface sign_ok=False.",
      },
      {
        q: "Why is elasticity reported both positive and negative in different views?",
        a: "Internally we store the signed coefficient (negative for downward-sloping demand). UI sometimes shows magnitude; check the field name.",
      },
      {
        q: "Why does the test set look so small?",
        a: "Chronological 80/20 holdout. Short PPGs get 4–5 test weeks. Larger panels naturally get wider hold-outs.",
      },
      {
        q: "What's hierarchical shrinkage?",
        a: "Partial pooling toward the category mean elasticity. Noisy PPG estimates shrink toward the cohort; clean ones move less. Reduces over-fitting.",
      },
      {
        q: "Why is my elasticity exactly -1.0?",
        a: "Coincidence or a thin fit. Check the std_err — wide intervals point to a poorly identified coefficient.",
      },
    ],
    cornerCases: [
      {
        condition: "Fewer than 20 weeks per PPG",
        behaviour: "PPG skipped; no contribution to downstream stages.",
      },
      {
        condition: "Wrong-sign log-log AND wrong-sign semi-log",
        behaviour: "LightGBM may still win on WAPE; sign_ok=False is surfaced.",
      },
      {
        condition: "Only log_price as a feature",
        behaviour: "Minimum viable; fit still runs.",
      },
      {
        condition: "LightGBM std_err derived from numerical perturbation",
        behaviour: "Excluded from hierarchical pooling (not a sampling SE).",
      },
      {
        condition: "Perfectly collinear features (rare after refine)",
        behaviour: "OLS produces NaN coefficients; LightGBM still fits.",
      },
      {
        condition: "All test rows have units = 0",
        behaviour: "WAPE undefined; PPG fails downstream verdict.",
      },
      {
        condition: "NaN in any feature row",
        behaviour: "Row dropped before fitting.",
      },
      {
        condition: "LLM unreachable",
        behaviour: "Empty narrative; model fits and verdicts unchanged.",
      },
    ],
    engineeringNotes: [
      { topic: "Train/test split", ref: "chronological 80/20" },
      { topic: "Minimum rows", ref: "20 observations per PPG" },
      { topic: "Sign-retry semi-log", ref: "core/agents/modeling.py:98-121" },
      { topic: "Approval gate", ref: "core/orchestrator/state.py" },
      { topic: "Per-agent model override", ref: "MODEL_MODELING env" },
    ],
  },

  results_reasoning: {
    docsAnchor: "results-reasoning",
    questions: [
      {
        q: "What makes a PPG 'pass'?",
        a: "All four checks clear: sign_ok=True, magnitude in [0.3, 6.0], R² ≥ 0.30, WAPE ≤ 0.30.",
      },
      {
        q: "Why is my PPG 'warn'?",
        a: "One or more thresholds breached but sign is correct. Magnitude outside band, R² below 0.30, or WAPE > 0.30 but < some fatal level.",
      },
      {
        q: "Why is 'fail' rare?",
        a: "It triggers when sign is wrong or the PPG was skipped at modeling. Most live PPGs land in pass or warn.",
      },
      {
        q: "What is the magnitude band [0.3, 6.0]?",
        a: "Typical CPG own-price elasticity range from published literature. Outside this band, results are suspicious and surfaced as warn.",
      },
      {
        q: "R² ≥ 0.30 seems low. Why?",
        a: "Panel-data noise is high (store-level seasonality, week-to-week swings). 0.30 means the model explains roughly a third of variance — a respectable floor.",
      },
      {
        q: "What if R² is 0.95?",
        a: "Passes here, but check the holdout WAPE in validation — over-fitting will inflate in-sample R².",
      },
      {
        q: "Does this stage refit anything?",
        a: "No. It consumes modeling artifacts and produces verdicts only.",
      },
      {
        q: "How do verdicts feed downstream?",
        a: "Surfaced in the insights summary. They do not gate optimization, but they colour the recommendation confidence.",
      },
    ],
    cornerCases: [
      {
        condition: "PPG skipped at modeling",
        behaviour: "Automatic fail at results_reasoning.",
      },
      {
        condition: "WAPE > 0.30 but sign and magnitude are fine",
        behaviour: "Verdict: warn (not fail).",
      },
      {
        condition: "Sign wrong but magnitude in band",
        behaviour: "Verdict: fail. Sign-correctness dominates.",
      },
      {
        condition: "R² = 0 (no variance explained)",
        behaviour: "Verdict: fail.",
      },
      {
        condition: "Magnitude exactly 0.30",
        behaviour: "Passes (inclusive lower bound).",
      },
      {
        condition: "LightGBM winner with no signed elasticity",
        behaviour: "Surfaced as 'no signed estimate'; sign check skipped, others apply.",
      },
      {
        condition: "All PPGs fail",
        behaviour: "Insights shows a red banner; no recommendations produced.",
      },
      {
        condition: "LLM unreachable",
        behaviour: "Verdicts deterministic; narrative blank.",
      },
    ],
    engineeringNotes: [
      { topic: "Thresholds", ref: "core/config.py ValidationThresholds" },
      { topic: "Override knobs", ref: "VALIDATION__SIGN_PASS, VALIDATION__WAPE_PASS env vars" },
      { topic: "Verdict logic", ref: "worst-status precedence: fail > warn > pass" },
    ],
  },

  decomposition: {
    docsAnchor: "decomposition",
    questions: [
      {
        q: "Why don't the components sum exactly to observed units?",
        a: "Residual term absorbs un-modelled variance. Reconciliation tolerance is reported; large mismatches drop confidence.",
      },
      {
        q: "What is 'base'?",
        a: "Predicted units when all drivers sit at reference values: no promo, mean ACV, mean competitor price, etc.",
      },
      {
        q: "Why is the price contribution sometimes positive?",
        a: "Price below the reference creates a lift (positive contribution). Price above creates drag (negative).",
      },
      {
        q: "Promo contribution for non-promo weeks is zero — is that right?",
        a: "Yes. We use group-wise ablation: turn off promo features and see the unit delta. Off-weeks already have zero, so no delta.",
      },
      {
        q: "Is decomposition per-PPG or panel?",
        a: "Per-PPG. Each winner is refit on the full series (no holdout) and decomposed at PPG×week grain.",
      },
      {
        q: "LightGBM decomposition feels coarse compared to OLS — why?",
        a: "LightGBM uses group-wise ablation, not closed-form coefficients. No per-feature granularity; only per-group totals.",
      },
      {
        q: "Why is seasonality flat?",
        a: "Only fires if week-of-year features survived refine. If they were pruned, the seasonality group is absent.",
      },
      {
        q: "Can I trust the absolute magnitudes?",
        a: "Within a PPG, yes — proportions are consistent. Across PPGs the absolute scale tracks the model's R².",
      },
    ],
    cornerCases: [
      {
        condition: "Reconciliation error > 1%",
        behaviour: "Confidence drops toward zero; surfaced in the audit table.",
      },
      {
        condition: "Non-OLS / non-LightGBM model",
        behaviour: "Skipped at decomposition; no contribution rows produced.",
      },
      {
        condition: "Single-week PPG (degenerate)",
        behaviour: "Trivial decomposition: only base, no driver groups.",
      },
      {
        condition: "Negative base value",
        behaviour: "Model artifact; surfaced as an anomaly in the table.",
      },
      {
        condition: "All weeks were promo",
        behaviour: "Base inherits the promo state; price contribution diluted.",
      },
      {
        condition: "Missing driver group (e.g. no competitor price)",
        behaviour: "Group entry is 0, not NaN. UI shows 0 contribution.",
      },
      {
        condition: "PPG skipped at modeling",
        behaviour: "Not decomposed.",
      },
      {
        condition: "LLM unreachable",
        behaviour: "Headline narrative falls back to template.",
      },
    ],
    engineeringNotes: [
      { topic: "OLS path (closed-form)", ref: "core/agents/decomposition.py" },
      { topic: "LightGBM path (ablation)", ref: "group-wise feature holdout" },
      { topic: "Reconciliation tolerance", ref: "1% target" },
    ],
  },

  simulation: {
    docsAnchor: "simulation",
    questions: [
      {
        q: "What price grid is swept?",
        a: "13 multipliers (e.g. 0.85 to 1.15 in steps) × 2 promo states (on/off) = 26 cells per PPG.",
      },
      {
        q: "Why are out-of-range predictions flat?",
        a: "LightGBM clamps at the training-price envelope. Extrapolating outside observed prices would be unreliable.",
      },
      {
        q: "Can I sweep more than promo on/off?",
        a: "Not yet. Depth, duration, and lift modelling are on the roadmap.",
      },
      {
        q: "What is the 'base price' here?",
        a: "Median observed price per PPG. LightGBM also reads a `log_base_price` feature where available.",
      },
      {
        q: "Does simulation respect competitor prices?",
        a: "Competitor price is held at the mean during the sweep. We don't co-vary it.",
      },
      {
        q: "Why does the heatmap look weird at extremes?",
        a: "Predictions flatten at the price envelope edges. The optimizer clips those rungs from the ladder.",
      },
      {
        q: "Does simulation use train or full data?",
        a: "Full data, same refit as decomposition. No holdout.",
      },
      {
        q: "Can I export the grid?",
        a: "Yes — `simulation_grid.json` is downloadable from the artifact list.",
      },
    ],
    cornerCases: [
      {
        condition: "PPG skipped at modeling",
        behaviour: "No simulation grid produced.",
      },
      {
        condition: "All features held at mean for LightGBM sweep",
        behaviour: "Interaction effects may look unrealistic; UI flags low confidence.",
      },
      {
        condition: "Multiplier × base price below cost",
        behaviour: "Cell still simulated; optimization will floor it on margin.",
      },
      {
        condition: "Promo=on but PPG never had promo history",
        behaviour: "Prediction is extrapolated blindly; surface as low-confidence cell.",
      },
      {
        condition: "Competitor price column entirely null",
        behaviour: "Competitor coefficient dropped; sweep proceeds without it.",
      },
      {
        condition: "Very narrow observed price range",
        behaviour: "Grid resolution stays at 13 multipliers; cells outside envelope are clipped at optimization.",
      },
      {
        condition: "NaN in context features",
        behaviour: "Forward-filled or replaced with mean before sweep.",
      },
      {
        condition: "simulation_summary smaller than simulation_grid",
        behaviour: "Expected — summary is the highlights table; grid is the full sweep.",
      },
    ],
    engineeringNotes: [
      { topic: "Grid shape", ref: "13 multipliers × 2 promo states = 26 cells per PPG" },
      { topic: "Envelope clipping", ref: "core/agents/optimization.py:169-247 (applied at optimization stage)" },
      { topic: "OLS path (closed-form)", ref: "core/agents/simulation.py" },
    ],
  },

  optimization: {
    docsAnchor: "optimization",
    questions: [
      {
        q: "Why does my recommended price differ from the continuous optimum?",
        a: "We enforce a discrete price ladder. The MILP picks the best ladder rung, which may step away from the continuous optimum.",
      },
      {
        q: "What is the margin floor?",
        a: "Default: 5% above cost. Configurable per run via the constraint editor in the approval gate.",
      },
      {
        q: "Can I customize the price ladder?",
        a: "Yes. Edit ladder values in the optimization approval gate and rerun the stage. The orchestrator re-solves without re-modelling.",
      },
      {
        q: "What does 'relaxed' mean?",
        a: "Strict MILP was infeasible. We re-solve with soft constraints, surface which constraints bound, and report the trade-off.",
      },
      {
        q: "How do I find which constraints conflicted?",
        a: "Check `optimization_constraints.json` artifact — the `binding_violations` array lists each.",
      },
      {
        q: "Why didn't price go to the cheapest ladder rung?",
        a: "Likely the margin floor or competitive gap blocked it. Inspect those constraints.",
      },
      {
        q: "What's 'envelope clipped'?",
        a: "LightGBM-only safeguard: ladder rungs outside the training price range are dropped before solving. Prevents extrapolation.",
      },
      {
        q: "Can I optimize for revenue instead of margin?",
        a: "Yes, set `options.objective='revenue'` at rerun time. Default is margin.",
      },
    ],
    cornerCases: [
      {
        condition: "Margin floor exceeds the only ladder rung satisfying comp gap",
        behaviour: "Infeasible MILP; soft-relaxation fires; binding violations surfaced.",
      },
      {
        condition: "Cost data missing",
        behaviour: "Fallback: 55% of base price assumed as cost.",
      },
      {
        condition: "Single-rung ladder",
        behaviour: "Trivially solved — only one feasible option.",
      },
      {
        condition: "LightGBM with all rungs outside training envelope",
        behaviour: "Keep full ladder; expect low confidence and rely on validation verdict.",
      },
      {
        condition: "PPG skipped at modeling",
        behaviour: "Not optimized; no recommendation produced.",
      },
      {
        condition: "Move guardrail conflicts with margin floor",
        behaviour: "Relax fires; binding violation reported.",
      },
      {
        condition: "Identical ladders across PPGs",
        behaviour: "Same constraints applied; OK — no special handling.",
      },
      {
        condition: "Rerun without changing options",
        behaviour: "Idempotent — same result.",
      },
    ],
    engineeringNotes: [
      { topic: "MILP solver", ref: "core/optimization/milp.py:130-242 (PuLP)" },
      { topic: "Envelope clipping", ref: "core/agents/optimization.py:169-247" },
      { topic: "Rerun support", ref: "POST /runs/{id}/rerun with options.optimization" },
      { topic: "Default cost", ref: "55% of base price" },
      { topic: "Approval gate", ref: "core/orchestrator/state.py APPROVAL_GATES" },
    ],
  },

  validation: {
    docsAnchor: "validation",
    questions: [
      {
        q: "What is rolling-origin cross-validation?",
        a: "Train set expands; test set slides forward in time. Four folds by default. Mimics how the model will be used in production.",
      },
      {
        q: "Why does my sign stability score change on rerun?",
        a: "It shouldn't — folds are deterministic given the data. If your data changed (new weeks added), folds shift.",
      },
      {
        q: "What is elasticity CV?",
        a: "Coefficient of variation of the elasticity estimate across folds. CV ≤ 0.40 indicates a stable estimate.",
      },
      {
        q: "Why is mean WAPE different from the modeling-stage WAPE?",
        a: "Modeling uses a single 80/20 split. Validation uses four sliding folds; the average is over multiple holdouts.",
      },
      {
        q: "Can I trust a 'warn' PPG for pricing decisions?",
        a: "With caveats. Inspect which check warned — wide elasticity CV is more concerning than a small magnitude excursion.",
      },
      {
        q: "What does 'magnitude check fails but sign and WAPE pass' mean?",
        a: "Mean absolute elasticity lies outside [0.3, 6.0]. Either too inelastic or too elastic for typical CPG.",
      },
      {
        q: "Why are residuals shown?",
        a: "Diagnostic. Look for non-random patterns (trends, fan-out) that suggest mis-specification.",
      },
      {
        q: "Does validation refit the winner model?",
        a: "Yes — same family, refit per fold. Hyperparameters carry over.",
      },
    ],
    cornerCases: [
      {
        condition: "Fewer than 4 fold-points available",
        behaviour: "Fewer folds used; n_folds reported.",
      },
      {
        condition: "Single fold pass",
        behaviour: "Sign stability check is degenerate (only 1 fold).",
      },
      {
        condition: "All folds fail",
        behaviour: "Verdict: fail; PPG flagged unusable.",
      },
      {
        condition: "Wrong sign in only one fold",
        behaviour: "Sign stability may still pass if ≥75% folds correct.",
      },
      {
        condition: "LightGBM in validation",
        behaviour: "No closed-form signed elasticity; sign stability derived via local decomposition.",
      },
      {
        condition: "PPG skipped at modeling",
        behaviour: "Not validated.",
      },
      {
        condition: "Residuals identically zero",
        behaviour: "Over-fit signal; surfaces but does not fail directly.",
      },
      {
        condition: "LLM unreachable",
        behaviour: "Numerical verdicts unchanged; narrative blank.",
      },
    ],
    engineeringNotes: [
      { topic: "Rolling-origin CV", ref: "core/agents/validation.py" },
      { topic: "Default folds", ref: "4 (configurable)" },
      { topic: "Thresholds (shared with results_reasoning)", ref: "core/config.py ValidationThresholds" },
    ],
  },

  insights: {
    docsAnchor: "insights",
    questions: [
      {
        q: "Where do the numbers in the report come from?",
        a: "Every figure traces back to an upstream artifact. Insights does not refit or recompute — only assembles.",
      },
      {
        q: "Why is the PDF missing?",
        a: "WeasyPrint failed to render (missing system fonts, library not installed, etc.). HTML is still available.",
      },
      {
        q: "What is 'total revenue uplift'?",
        a: "Sum of revenue delta across optimized PPGs, comparing recommended prices to baseline prices.",
      },
      {
        q: "Can I filter the report by category or brand?",
        a: "Not yet via UI. Filter the underlying artifacts post-run.",
      },
      {
        q: "Why are some PPGs missing from the report?",
        a: "Skipped upstream — either ineligible at PPG selection or unmodelled due to thin data.",
      },
      {
        q: "What does the overall confidence score mean?",
        a: "n_pass / n_validated. Drops to zero if validation didn't run for any PPG.",
      },
      {
        q: "Why does the headline narrative use 'approximately'?",
        a: "When the LLM is in dry-run mode (no API key), narrative comes from a deterministic template that hedges intentionally.",
      },
      {
        q: "Where's the cost dashboard?",
        a: "Separate artifact `cost_summary.json`. Rendered in the run sidebar; not embedded in the PDF.",
      },
    ],
    cornerCases: [
      {
        condition: "All PPGs failed verdict",
        behaviour: "Red banner; no recommendations section.",
      },
      {
        condition: "WeasyPrint missing on host",
        behaviour: "PDF artifact absent; HTML still served.",
      },
      {
        condition: "Validation stage skipped entirely",
        behaviour: "Confidence score falls back to 0.",
      },
      {
        condition: "No optimization results",
        behaviour: "Recommendations section blank; headline still shows.",
      },
      {
        condition: "Run rerun after editing constraints",
        behaviour: "Numbers refresh; report regenerated.",
      },
      {
        condition: "Live card numbers differ from report",
        behaviour: "Cards are live; report is a snapshot at insights-run time.",
      },
      {
        condition: "Large run (>20 PPGs)",
        behaviour: "Tables paginate in HTML; PDF flows naturally across pages.",
      },
      {
        condition: "LLM unreachable",
        behaviour: "Headline uses deterministic template; metrics unchanged.",
      },
    ],
    engineeringNotes: [
      { topic: "Assembly only", ref: "no refit — reads upstream artifacts" },
      { topic: "PDF renderer", ref: "WeasyPrint (optional dep)" },
      { topic: "Confidence formula", ref: "n_pass / n_validated" },
      { topic: "Cost summary", ref: "cost_summary.json" },
    ],
  },
};
