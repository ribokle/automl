import type { AgentName, AgentStatus } from "./types";

export const AGENT_META: Record<AgentName, { title: string; description: string }> = {
  ingestion: {
    title: "Ingestion",
    description:
      "Load the CSV into DuckDB, build the dbt panel mart, run schema + distribution checks, profile each column.",
  },
  ppg_mapping: {
    title: "PPG Mapping",
    description:
      "Group SKUs into Price-Pack Groups by brand, category and price tier. Score each mapping by within-group price coherence.",
  },
  ppg_selection: {
    title: "PPG Selection",
    description:
      "Score each PPG on size, coverage, price variation and promo activity. Flag the ones eligible for modelling.",
  },
  feature_selection: {
    title: "Feature Selection",
    description: "Pick the candidate covariates for elasticity modelling.",
  },
  eda: {
    title: "EDA",
    description: "Summarise distributions and pairwise relationships at the PPG × week level.",
  },
  feature_engineering: {
    title: "Feature Engineering",
    description: "Build lagged, holiday and competitive-price features.",
  },
  feature_refine: {
    title: "Feature Refine",
    description: "Drop collinear features (VIF, |corr| pruning).",
  },
  modeling: {
    title: "Modeling",
    description: "Fit per-PPG log-log price-elasticity models.",
  },
  results_reasoning: {
    title: "Results Reasoning",
    description: "Narrate elasticity estimates, fit quality and which PPGs to trust.",
  },
  decomposition: {
    title: "Decomposition",
    description: "Decompose observed units into base / price / promo / seasonality drivers.",
  },
  simulation: {
    title: "Simulation",
    description: "Replay counterfactual price / promo scenarios through the fitted model.",
  },
  optimization: {
    title: "Optimization",
    description:
      "Solve for prices that maximise margin under ladder, margin-floor and competitor-gap constraints.",
  },
  validation: {
    title: "Validation",
    description: "Hold-out WAPE, residual diagnostics, sanity bounds on recommended moves.",
  },
  insights: {
    title: "Insights",
    description: "Render the executive HTML / PDF report and the cost dashboard.",
  },
};

export const STATUS_STYLE: Record<AgentStatus | "idle", { dot: string; pill: string; ring: string }> = {
  idle: {
    dot: "bg-slate-700",
    pill: "bg-slate-800 text-slate-400 border-slate-700",
    ring: "ring-slate-800",
  },
  pending: {
    dot: "bg-slate-700",
    pill: "bg-slate-800 text-slate-400 border-slate-700",
    ring: "ring-slate-800",
  },
  running: {
    dot: "bg-amber-400 animate-pulse",
    pill: "bg-amber-500/15 text-amber-300 border-amber-500/40",
    ring: "ring-amber-500/40",
  },
  awaiting_approval: {
    dot: "bg-purple-400 animate-pulse",
    pill: "bg-purple-500/15 text-purple-300 border-purple-500/40",
    ring: "ring-purple-500/40",
  },
  done: {
    dot: "bg-emerald-400",
    pill: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
    ring: "ring-emerald-500/30",
  },
  failed: {
    dot: "bg-rose-500",
    pill: "bg-rose-500/15 text-rose-300 border-rose-500/40",
    ring: "ring-rose-500/40",
  },
  skipped: {
    dot: "bg-slate-600",
    pill: "bg-slate-800/60 text-slate-500 border-slate-700",
    ring: "ring-slate-800",
  },
};

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function summariseOutputs(
  agent: AgentName,
  outputs: Record<string, unknown> | null | undefined,
): string[] {
  if (!outputs) return [];
  const o = outputs as Record<string, number | boolean | string | undefined>;
  if (o.mocked) return ["stub — implementation lands in a later phase"];
  switch (agent) {
    case "ingestion":
      return [
        typeof o.row_count === "number" ? `${formatNumber(o.row_count)} rows loaded` : "",
        typeof o.dbt_failures === "number" ? `${o.dbt_failures} dbt failures` : "",
        typeof o.ge_failures === "number" ? `${o.ge_failures} GE failures` : "",
        typeof o.n_anomalies === "number" ? `${o.n_anomalies} anomalies flagged` : "",
      ].filter(Boolean);
    case "ppg_mapping":
      return [
        typeof o.n_ppgs === "number" ? `${o.n_ppgs} PPGs` : "",
        typeof o.n_skus === "number" ? `${o.n_skus} SKUs grouped` : "",
        typeof o.n_flagged === "number" ? `${o.n_flagged} flagged` : "",
        typeof o.mean_confidence === "number" ? `mean conf ${Number(o.mean_confidence).toFixed(2)}` : "",
      ].filter(Boolean);
    case "ppg_selection":
      return [
        typeof o.n_ppgs === "number" ? `${o.n_ppgs} PPGs scored` : "",
        typeof o.n_eligible === "number" ? `${o.n_eligible} eligible for modelling` : "",
        typeof o.mean_score === "number" ? `mean score ${Number(o.mean_score).toFixed(2)}` : "",
      ].filter(Boolean);
    case "modeling":
      return [
        typeof o.n_total === "number" && typeof o.n_correct_sign === "number"
          ? `${o.n_correct_sign}/${o.n_total} correct sign`
          : "",
        typeof o.n_retries === "number" ? `${o.n_retries} semi-log retries` : "",
        typeof o.n_shap === "number" ? `${o.n_shap} SHAP summaries` : "",
        typeof o.n_shrunk === "number" && o.n_shrunk > 0
          ? `${o.n_shrunk} PPGs pooled · τ²=${Number(o.tau_squared ?? 0).toFixed(2)}`
          : "",
      ].filter(Boolean);
    case "optimization":
      return [
        typeof o.n_optimised === "number" ? `${o.n_optimised} PPGs optimised` : "",
        typeof o.objective === "string" ? `objective: ${o.objective}` : "",
        typeof o.ladder_size === "number" ? `${o.ladder_size}-rung ladder` : "",
        typeof o.n_relaxed === "number" && o.n_relaxed > 0
          ? `${o.n_relaxed} relaxed`
          : "",
      ].filter(Boolean);
    case "validation":
      return [
        typeof o.n_validated === "number" ? `${o.n_validated} PPGs validated` : "",
        typeof o.n_folds === "number" ? `${o.n_folds}-fold rolling CV` : "",
        typeof o.n_pass === "number" && typeof o.n_validated === "number"
          ? `${o.n_pass}/${o.n_validated} pass`
          : "",
        typeof o.n_fail === "number" && o.n_fail > 0 ? `${o.n_fail} fail` : "",
      ].filter(Boolean);
    default:
      return Object.entries(o)
        .slice(0, 4)
        .map(([k, v]) => `${k} = ${typeof v === "number" ? formatNumber(v) : String(v)}`);
  }
}

export function summariseTool(tool: string, ev: Record<string, unknown>): string {
  const bits: string[] = [];
  if (typeof ev.rows === "number") bits.push(`${formatNumber(ev.rows)} rows`);
  if (typeof ev.checks === "number") bits.push(`${ev.checks} checks`);
  if (typeof ev.columns === "number") bits.push(`${ev.columns} columns`);
  if (typeof ev.n_skus === "number") bits.push(`${ev.n_skus} SKUs`);
  if (typeof ev.n_ppgs === "number") bits.push(`${ev.n_ppgs} PPGs`);
  return bits.length ? `${tool} · ${bits.join(" · ")}` : tool;
}

export function formatDuration(start?: string | null, end?: string | null): string | null {
  if (!start) return null;
  const t0 = Date.parse(start);
  const t1 = end ? Date.parse(end) : Date.now();
  if (!Number.isFinite(t0) || !Number.isFinite(t1)) return null;
  const ms = Math.max(0, t1 - t0);
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}
