import { STATUS_STYLE } from "./theme";
import type { AgentName, RunEvent } from "./types";

export { STATUS_STYLE };

interface AgentMetaEntry {
  title: string;
  description: string;
  hasVisuals: boolean;
  hasLLM: boolean;
}

export const AGENT_META: Record<AgentName, AgentMetaEntry> = {
  ingestion: {
    title: "Ingestion",
    description:
      "Load the CSV into DuckDB, build the dbt panel mart, run schema + distribution checks, profile each column.",
    hasVisuals: true,
    hasLLM: true,
  },
  ppg_mapping: {
    title: "PPG Mapping",
    description:
      "Group SKUs into Price-Pack Groups by brand, category and price tier. Score each mapping by within-group price coherence.",
    hasVisuals: true,
    hasLLM: true,
  },
  ppg_selection: {
    title: "PPG Selection",
    description:
      "Score each PPG on size, coverage, price variation and promo activity. Flag the ones eligible for modelling.",
    hasVisuals: true,
    hasLLM: true,
  },
  feature_selection: {
    title: "Feature Selection",
    description: "Pick the candidate covariates for elasticity modelling.",
    hasVisuals: false,
    hasLLM: true,
  },
  eda: {
    title: "EDA",
    description: "Summarise distributions and pairwise relationships at the PPG × week level.",
    hasVisuals: true,
    hasLLM: true,
  },
  advanced_eda: {
    title: "Advanced EDA",
    description:
      "Time-series decomposition, structural anomaly detection, change points, promo lift sketches, cross-PPG correlation, Pareto / ABC, and a promo calendar.",
    hasVisuals: true,
    hasLLM: true,
  },
  feature_engineering: {
    title: "Feature Engineering",
    description: "Build lagged, holiday and competitive-price features.",
    hasVisuals: true,
    hasLLM: true,
  },
  feature_refine: {
    title: "Feature Refine",
    description: "Drop collinear features (VIF, |corr| pruning).",
    hasVisuals: true,
    hasLLM: true,
  },
  modeling: {
    title: "Modeling",
    description: "Fit per-PPG log-log price-elasticity models.",
    hasVisuals: true,
    hasLLM: true,
  },
  results_reasoning: {
    title: "Results Reasoning",
    description: "Narrate elasticity estimates, fit quality and which PPGs to trust.",
    hasVisuals: false,
    hasLLM: false,
  },
  decomposition: {
    title: "Decomposition",
    description: "Decompose observed units into base / price / promo / seasonality drivers.",
    hasVisuals: true,
    hasLLM: false,
  },
  simulation: {
    title: "Simulation",
    description: "Replay counterfactual price / promo scenarios through the fitted model.",
    hasVisuals: true,
    hasLLM: false,
  },
  optimization: {
    title: "Optimization",
    description:
      "Solve for prices that maximise margin under ladder, margin-floor and competitor-gap constraints.",
    hasVisuals: true,
    hasLLM: true,
  },
  validation: {
    title: "Validation",
    description: "Hold-out WAPE, residual diagnostics, sanity bounds on recommended moves.",
    hasVisuals: true,
    hasLLM: true,
  },
  insights: {
    title: "Insights",
    description: "Render the executive HTML / PDF report and the cost dashboard.",
    hasVisuals: true,
    hasLLM: true,
  },
};

const _entries = Object.entries(AGENT_META) as [AgentName, AgentMetaEntry][];
export const VISUALS_AGENTS: ReadonlySet<AgentName> = new Set(
  _entries.filter(([, m]) => m.hasVisuals).map(([k]) => k),
);
export const LLM_AGENTS: ReadonlySet<AgentName> = new Set(
  _entries.filter(([, m]) => m.hasLLM).map(([k]) => k),
);

export const formatDisplayName = (s: string): string => s.replace(/_/g, " ");

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
        typeof o.n_envelope_clipped === "number" && o.n_envelope_clipped > 0
          ? `${o.n_envelope_clipped} envelope-clipped`
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
    case "advanced_eda":
      return [
        typeof o.n_ppgs_top_k === "number" ? `${o.n_ppgs_top_k} PPGs analysed` : "",
        typeof o.n_anomalies === "number" ? `${o.n_anomalies} anomalies` : "",
        typeof o.n_change_points === "number" ? `${o.n_change_points} change points` : "",
        typeof o.stationarity_pass_rate === "number"
          ? `${Math.round(Number(o.stationarity_pass_rate) * 100)}% stationary`
          : "",
      ].filter(Boolean);
    case "insights":
      return [
        typeof o.n_ppgs === "number" ? `${o.n_ppgs} PPGs reported` : "",
        typeof o.total_revenue === "number"
          ? `revenue $${formatNumber(Number(o.total_revenue))}`
          : "",
        typeof o.total_margin === "number"
          ? `margin $${formatNumber(Number(o.total_margin))}`
          : "",
        o.pdf === true ? "HTML + PDF" : o.pdf === false ? "HTML (PDF failed)" : "",
      ].filter(Boolean);
    default:
      return Object.entries(o)
        .slice(0, 4)
        .map(([k, v]) => `${k} = ${typeof v === "number" ? formatNumber(v) : String(v)}`);
  }
}

export function summariseTool(tool: string, ev: RunEvent): string {
  const bits: string[] = [];
  if (typeof ev.rows === "number") bits.push(`${formatNumber(ev.rows)} rows`);
  if (typeof ev.checks === "number") bits.push(`${ev.checks} checks`);
  if (typeof ev.columns === "number") bits.push(`${ev.columns} columns`);
  if (typeof ev.n_skus === "number") bits.push(`${ev.n_skus} SKUs`);
  if (typeof ev.n_ppgs === "number") bits.push(`${ev.n_ppgs} PPGs`);
  return bits.length ? `${tool} · ${bits.join(" · ")}` : tool;
}

export function fmtClock(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleTimeString();
}

export function relativeTime(iso: string): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "—";
  const diff = (Date.now() - t) / 1000;
  if (diff < 60) return `${Math.round(diff)}s`;
  if (diff < 3600) return `${Math.round(diff / 60)}m`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h`;
  return `${Math.round(diff / 86400)}d`;
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
