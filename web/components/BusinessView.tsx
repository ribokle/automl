"use client";

/**
 * Business-focused summary of a completed run.
 *
 * Surfaces the executive headline, aggregated KPIs, recommended actions
 * per PPG, and model quality in a presentation-ready layout. Designed
 * for a CPG operator or category manager who wants decisions, not
 * diagnostics.
 */

import { useEffect, useState } from "react";
import { artifactUrl, getArtifact } from "@/lib/api";

/* ------------------------------------------------------------------ */
/*  Types (keep minimal — only what the UI actually renders)           */
/* ------------------------------------------------------------------ */

interface Kpis {
  n_optimised?: number;
  n_feasible?: number;
  n_relaxed?: number;
  n_pass?: number;
  n_warn?: number;
  n_fail?: number;
  total_revenue?: number;
  total_margin?: number;
}

interface PerPPG {
  ppg_id: string;
  recommended_price?: number;
  price_multiplier?: number;
  promo?: boolean | number;
  units?: number;
  revenue?: number;
  margin?: number;
  relaxed?: boolean;
  verdict?: string;
  elasticity?: number;
  rationale?: string;
}

interface InsightsSummary {
  headline?: string;
  kpis?: Kpis;
  per_ppg?: PerPPG[];
  validation?: Record<string, unknown>;
  objective?: string;
}

interface ValidationRow {
  ppg_id: string;
  category?: string;
  verdict?: string;
  wape_mean?: number;
  elasticity_mean?: number;
  benchmark_status?: string;
}

interface ElasticityRow {
  ppg_id: string;
  elasticity?: number;
  n_models?: number;
}

/* ------------------------------------------------------------------ */
/*  Small presentational helpers                                        */
/* ------------------------------------------------------------------ */

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex flex-col gap-0.5 rounded border border-slate-800 bg-slate-900/40 px-4 py-3">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">{label}</span>
      <span className="font-mono text-xl text-emerald-300">{value}</span>
      {sub && <span className="text-[10px] text-slate-500">{sub}</span>}
    </div>
  );
}

function verdictColor(v?: string) {
  if (!v) return "text-slate-400";
  if (v.toLowerCase().includes("pass")) return "text-emerald-300";
  if (v.toLowerCase().includes("warn")) return "text-amber-300";
  return "text-rose-300";
}

function benchmarkColor(s?: string) {
  if (!s) return "text-slate-400";
  if (s === "in_range") return "text-emerald-300";
  if (s === "too_low" || s === "too_high") return "text-amber-300";
  return "text-slate-400";
}

function fmt$(n?: number) {
  if (n == null) return "—";
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function fmtPct(n?: number) {
  if (n == null) return "—";
  return `${(n * 100).toFixed(1)}%`;
}

function fmtElasticity(n?: number) {
  if (n == null) return "—";
  return n.toFixed(2);
}

/* ------------------------------------------------------------------ */
/*  Main component                                                       */
/* ------------------------------------------------------------------ */

export function BusinessView({ runId }: { runId: string }) {
  const [summary, setSummary] = useState<InsightsSummary | null>(null);
  const [valRows, setValRows] = useState<ValidationRow[]>([]);
  const [elasticities, setElasticities] = useState<ElasticityRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const [s, v, e] = await Promise.all([
        getArtifact<InsightsSummary>(runId, "insights_summary.json").catch(() => null),
        getArtifact<ValidationRow[]>(runId, "validation_table.json").catch(() => null),
        getArtifact<ElasticityRow[]>(runId, "elasticity_per_ppg.json").catch(() => null),
      ]);
      if (!cancelled) {
        setSummary(s);
        setValRows(Array.isArray(v) ? v : []);
        setElasticities(Array.isArray(e) ? e : []);
        setLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [runId]);

  if (loading) {
    return (
      <div className="py-12 text-center text-[12px] text-slate-500">Loading business view…</div>
    );
  }

  if (!summary) {
    return (
      <div className="rounded border border-slate-800 bg-slate-900/40 px-4 py-8 text-center text-[12px] text-slate-400">
        Business view is available after the{" "}
        <code className="rounded bg-slate-800 px-1 font-mono text-[10px]">insights</code> agent
        completes.
      </div>
    );
  }

  const kpis = summary.kpis ?? {};
  const perPpg = summary.per_ppg ?? [];
  // Derive text recommendations from per-PPG rationale strings.
  // (summary.recommendations is opt_table rows, not strings.)
  const recommendations = perPpg
    .filter((r) => r.rationale)
    .map((r) => `${r.ppg_id}: ${r.rationale}`);

  // Sort per-PPG by revenue descending for the table.
  const sorted = [...perPpg].sort((a, b) => (b.revenue ?? 0) - (a.revenue ?? 0));

  // Top / bottom 5 elasticities.
  const elasticityRanked = [...elasticities]
    .filter((r) => r.elasticity != null)
    .sort((a, b) => (a.elasticity ?? 0) - (b.elasticity ?? 0));

  const verdictCounts: Record<string, number> = {};
  for (const r of valRows) {
    const v = r.verdict ?? "unknown";
    verdictCounts[v] = (verdictCounts[v] ?? 0) + 1;
  }

  return (
    <div className="space-y-8">
      {/* Executive headline */}
      {summary.headline && (
        <div className="rounded border border-emerald-500/30 bg-emerald-500/5 px-4 py-3">
          <p className="text-sm text-emerald-200">{summary.headline}</p>
          {summary.objective && (
            <p className="mt-1 text-[10px] uppercase tracking-wider text-emerald-400/70">
              Objective: {summary.objective}
            </p>
          )}
        </div>
      )}

      {/* KPI row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Optimised PPGs" value={String(kpis.n_optimised ?? perPpg.length)} />
        <Stat
          label="Projected revenue"
          value={fmt$(kpis.total_revenue)}
          sub="sum of PPG revenue"
        />
        <Stat
          label="Projected margin"
          value={fmt$(kpis.total_margin)}
          sub="sum of PPG margin"
        />
        <Stat
          label="Validation"
          value={`${kpis.n_pass ?? 0} / ${(kpis.n_pass ?? 0) + (kpis.n_warn ?? 0) + (kpis.n_fail ?? 0)} pass`}
          sub={kpis.n_fail ? `${kpis.n_fail} below benchmark` : "all within benchmarks"}
        />
      </div>

      {/* Report downloads */}
      <div className="flex flex-wrap gap-2">
        <a
          href={artifactUrl(runId, "report.html")}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 rounded border border-slate-700 bg-slate-900/60 px-3 py-1.5 text-[11px] text-slate-300 hover:bg-slate-800"
        >
          ↓ HTML report
        </a>
        <a
          href={artifactUrl(runId, "report.pdf")}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 rounded border border-slate-700 bg-slate-900/60 px-3 py-1.5 text-[11px] text-slate-300 hover:bg-slate-800"
        >
          ↓ PDF report
        </a>
      </div>

      {/* Price recommendations table */}
      {sorted.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            Price recommendations · {sorted.length} PPGs
          </h2>
          <div className="overflow-x-auto rounded border border-slate-800">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="border-b border-slate-800 text-left text-[9.5px] uppercase tracking-wider text-slate-500">
                  <th className="px-3 py-2">PPG</th>
                  <th className="px-3 py-2">Rec. price</th>
                  <th className="px-3 py-2">Change</th>
                  <th className="px-3 py-2">Promo</th>
                  <th className="px-3 py-2">Revenue</th>
                  <th className="px-3 py-2">Margin</th>
                  <th className="px-3 py-2">Elasticity</th>
                  <th className="px-3 py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((r) => {
                  const change =
                    r.price_multiplier != null
                      ? r.price_multiplier >= 1
                        ? `+${((r.price_multiplier - 1) * 100).toFixed(1)}%`
                        : `${((r.price_multiplier - 1) * 100).toFixed(1)}%`
                      : "—";
                  const changeColor =
                    r.price_multiplier == null
                      ? "text-slate-400"
                      : r.price_multiplier > 1
                        ? "text-emerald-300"
                        : r.price_multiplier < 1
                          ? "text-rose-300"
                          : "text-slate-400";
                  return (
                    <tr
                      key={r.ppg_id}
                      className="border-b border-slate-800/60 last:border-0 hover:bg-slate-900/30"
                    >
                      <td className="px-3 py-1.5 font-mono text-slate-200">{r.ppg_id}</td>
                      <td className="px-3 py-1.5 font-mono text-slate-200">
                        {r.recommended_price != null ? `$${r.recommended_price.toFixed(2)}` : "—"}
                      </td>
                      <td className={`px-3 py-1.5 font-mono font-semibold ${changeColor}`}>
                        {change}
                      </td>
                      <td className="px-3 py-1.5 text-slate-300">
                        {r.promo ? "yes" : "—"}
                      </td>
                      <td className="px-3 py-1.5 font-mono text-slate-200">
                        {fmt$(r.revenue)}
                      </td>
                      <td className="px-3 py-1.5 font-mono text-slate-200">
                        {fmt$(r.margin)}
                      </td>
                      <td className="px-3 py-1.5 font-mono text-slate-400">
                        {fmtElasticity(r.elasticity)}
                      </td>
                      <td className={`px-3 py-1.5 ${r.relaxed ? "text-amber-300" : "text-slate-400"}`}>
                        {r.relaxed ? "relaxed" : "optimal"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Validation summary */}
        {valRows.length > 0 && (
          <section className="space-y-2">
            <h2 className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Model quality · {valRows.length} PPGs
            </h2>
            <div className="overflow-x-auto rounded border border-slate-800">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="border-b border-slate-800 text-left text-[9.5px] uppercase tracking-wider text-slate-500">
                    <th className="px-3 py-2">PPG</th>
                    <th className="px-3 py-2">Verdict</th>
                    <th className="px-3 py-2">WAPE</th>
                    <th className="px-3 py-2">Benchmark</th>
                  </tr>
                </thead>
                <tbody>
                  {valRows.map((r) => (
                    <tr
                      key={r.ppg_id}
                      className="border-b border-slate-800/60 last:border-0 hover:bg-slate-900/30"
                    >
                      <td className="px-3 py-1.5 font-mono text-slate-200">{r.ppg_id}</td>
                      <td className={`px-3 py-1.5 font-semibold ${verdictColor(r.verdict)}`}>
                        {r.verdict ?? "—"}
                      </td>
                      <td className="px-3 py-1.5 font-mono text-slate-300">
                        {r.wape_mean != null ? fmtPct(r.wape_mean) : "—"}
                      </td>
                      <td className={`px-3 py-1.5 ${benchmarkColor(r.benchmark_status)}`}>
                        {r.benchmark_status?.replace(/_/g, " ") ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Elasticity ranking */}
        {elasticityRanked.length > 0 && (
          <section className="space-y-2">
            <h2 className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Price sensitivity ranking
            </h2>
            <p className="text-[10px] text-slate-500">
              More negative = more price-sensitive. Range shows IQR across stores / folds.
            </p>
            <div className="space-y-1">
              {elasticityRanked.slice(0, 10).map((r) => {
                const e = r.elasticity ?? 0;
                const maxAbs = Math.max(
                  ...elasticityRanked.slice(0, 10).map((x) => Math.abs(x.elasticity ?? 0)),
                  0.1,
                );
                const barWidth = `${Math.round((Math.abs(e) / maxAbs) * 100)}%`;
                return (
                  <div key={r.ppg_id} className="flex items-center gap-2 text-[11px]">
                    <span className="w-28 shrink-0 truncate font-mono text-slate-300">
                      {r.ppg_id}
                    </span>
                    <div className="flex-1 rounded-full bg-slate-800">
                      <div
                        className="h-2 rounded-full bg-sky-500/60"
                        style={{ width: barWidth }}
                      />
                    </div>
                    <span className="w-14 text-right font-mono text-sky-300">
                      {fmtElasticity(e)}
                    </span>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </div>

      {/* Key recommendations */}
      {recommendations.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            Top recommendations ({Math.min(recommendations.length, 5)} of {recommendations.length})
          </h2>
          <ul className="space-y-1.5">
            {recommendations.slice(0, 5).map((r, i) => (
              <li
                key={i}
                className="flex items-start gap-2 rounded border border-slate-800 bg-slate-900/40 px-3 py-2 text-[11px] text-slate-300"
              >
                <span className="mt-0.5 shrink-0 rounded bg-emerald-500/20 px-1.5 py-0.5 text-[9px] font-semibold text-emerald-300">
                  {i + 1}
                </span>
                {r}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
