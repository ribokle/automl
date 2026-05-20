"use client";

import { useEffect, useState } from "react";
import { artifactUrl, getArtifact } from "@/lib/api";
import type { InsightsSummaryBlob } from "./tables/InsightsSummary";

interface Props {
  runId: string;
  insightsReady: boolean;
  hasPdf: boolean;
}

function currency(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}k`;
  return `$${n.toFixed(2)}`;
}

export function ExecutiveBanner({ runId, insightsReady, hasPdf }: Props) {
  const [data, setData] = useState<InsightsSummaryBlob | null>(null);
  useEffect(() => {
    if (!insightsReady) return;
    let cancelled = false;
    getArtifact<InsightsSummaryBlob>(runId, "insights_summary.json")
      .then((d) => {
        if (!cancelled && d && !("missing_columns" in d)) setData(d);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [runId, insightsReady]);

  if (!insightsReady || !data) return null;
  const k = data.kpis;

  return (
    <section className="rounded-xl border border-emerald-500/30 bg-gradient-to-br from-emerald-500/[0.06] via-slate-900/40 to-slate-900/40 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 max-w-3xl">
          <p className="text-[10px] uppercase tracking-wider text-emerald-300/80">
            Executive summary
          </p>
          <p className="mt-1 text-sm text-slate-100">{data.headline || "No headline available."}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <a
            href={artifactUrl(runId, "report.html")}
            target="_blank"
            rel="noreferrer"
            className="rounded bg-emerald-500/20 px-3 py-1 text-xs font-medium text-emerald-200 hover:bg-emerald-500/30"
          >
            Open report ↗
          </a>
          {hasPdf && (
            <a
              href={artifactUrl(runId, "report.pdf")}
              target="_blank"
              rel="noreferrer"
              className="rounded border border-slate-700 bg-slate-900 px-3 py-1 text-xs font-medium text-slate-200 hover:bg-slate-800"
            >
              PDF ↗
            </a>
          )}
        </div>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Kpi label="PPGs optimised" value={`${k.n_optimised}`} />
        <Kpi label="Strict feasible" value={`${k.n_feasible} / ${k.n_optimised}`} tone="emerald" />
        <Kpi
          label="Validation pass"
          value={`${k.n_pass} / ${k.n_validated}`}
          tone={k.n_fail > 0 ? "warn" : "emerald"}
        />
        <Kpi label="Recommended revenue" value={currency(k.total_revenue)} tone="emerald" />
      </div>
    </section>
  );
}

function Kpi({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "emerald" | "warn";
}) {
  const ring =
    tone === "emerald"
      ? "border-emerald-500/30"
      : tone === "warn"
        ? "border-amber-500/30"
        : "border-slate-800";
  return (
    <div className={`rounded border ${ring} bg-slate-950/60 p-2`}>
      <div className="text-[9px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="mt-0.5 font-mono text-sm text-slate-100">{value}</div>
    </div>
  );
}
