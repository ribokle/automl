"use client";

import { artifactUrl } from "@/lib/api";

export interface InsightsSummaryBlob {
  run_id: string;
  generated_at: string;
  objective: string;
  headline: string;
  kpis: {
    n_optimised: number;
    n_feasible: number;
    n_relaxed: number;
    n_validated: number;
    n_pass: number;
    n_warn: number;
    n_fail: number;
    total_revenue: number;
    total_margin: number;
  };
  per_ppg: {
    ppg_id: string;
    recommended_price: number | null;
    price_multiplier: number;
    promo: number;
    revenue: number | null;
    margin: number | null;
    relaxed: boolean;
    verdict: string;
    elasticity: number | null;
    rationale: string;
  }[];
}

function currency(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}k`;
  return `$${n.toFixed(2)}`;
}

function signedPct(n: number): string {
  const sign = n >= 0 ? "+" : "";
  return `${sign}${(n * 100).toFixed(1)}%`;
}

const VERDICT_STYLE: Record<string, string> = {
  pass: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  warn: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  fail: "bg-rose-500/15 text-rose-300 border-rose-500/40",
};

export function InsightsSummary({
  runId,
  data,
  hasPdf,
}: {
  runId: string;
  data: InsightsSummaryBlob;
  hasPdf: boolean;
}) {
  const k = data.kpis;
  return (
    <div className="space-y-4">
      {data.headline && (
        <div className="rounded-md border-l-2 border-emerald-500/60 bg-slate-900/60 px-3 py-2 text-xs text-slate-200">
          {data.headline}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Kpi label="PPGs optimised" value={`${k.n_optimised}`} />
        <Kpi label="Strict feasible" value={`${k.n_feasible} / ${k.n_optimised}`} />
        <Kpi label="Validation pass" value={`${k.n_pass} / ${k.n_validated}`} />
        <Kpi label="Recommended revenue" value={currency(k.total_revenue)} />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <a
          href={artifactUrl(runId, "report.html")}
          target="_blank"
          rel="noreferrer"
          className="rounded bg-emerald-500/20 px-3 py-1 text-xs font-medium text-emerald-200 hover:bg-emerald-500/30"
        >
          Open HTML report ↗
        </a>
        {hasPdf && (
          <a
            href={artifactUrl(runId, "report.pdf")}
            target="_blank"
            rel="noreferrer"
            className="rounded border border-slate-700 bg-slate-900 px-3 py-1 text-xs font-medium text-slate-200 hover:bg-slate-800"
          >
            Download PDF ↗
          </a>
        )}
        <span className="text-[10px] text-slate-500">
          Generated {data.generated_at} · objective {data.objective}
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[11px]">
          <thead>
            <tr className="border-b border-slate-800 text-[10px] uppercase tracking-wider text-slate-500">
              <th className="py-1 pr-3">PPG</th>
              <th className="py-1 pr-3 text-right">Δ price</th>
              <th className="py-1 pr-3">Promo</th>
              <th className="py-1 pr-3 text-right">Revenue</th>
              <th className="py-1 pr-3 text-right">Margin</th>
              <th className="py-1 pr-3 text-right">ε̂</th>
              <th className="py-1 pr-3">Verdict</th>
              <th className="py-1 pr-3">Rationale</th>
            </tr>
          </thead>
          <tbody>
            {data.per_ppg.map((p) => {
              const d = p.price_multiplier - 1.0;
              return (
                <tr key={p.ppg_id} className="border-b border-slate-900 align-top">
                  <td className="py-1 pr-3 font-mono">{p.ppg_id}</td>
                  <td className={`py-1 pr-3 text-right font-mono ${d >= 0 ? "text-emerald-300" : "text-amber-300"}`}>
                    {signedPct(d)}
                  </td>
                  <td className="py-1 pr-3">{p.promo === 1 ? "on" : "off"}</td>
                  <td className="py-1 pr-3 text-right font-mono">{currency(p.revenue)}</td>
                  <td className="py-1 pr-3 text-right font-mono">{currency(p.margin)}</td>
                  <td className="py-1 pr-3 text-right font-mono">
                    {p.elasticity != null ? p.elasticity.toFixed(2) : "—"}
                  </td>
                  <td className="py-1 pr-3">
                    <span
                      className={`rounded border px-1.5 py-0.5 text-[9px] uppercase tracking-wider ${
                        VERDICT_STYLE[p.verdict] ?? VERDICT_STYLE.warn
                      }`}
                    >
                      {p.verdict}
                    </span>
                  </td>
                  <td className="py-1 pr-3 text-slate-300">{p.rationale}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-slate-800 bg-slate-900/40 p-2">
      <div className="text-[9px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="mt-0.5 font-mono text-sm text-slate-100">{value}</div>
    </div>
  );
}
