"use client";

import { EChart } from "./EChart";

interface Hist {
  column: string;
  counts: number[];
  edges: number[];
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  n: number;
}

export interface HistogramsData {
  bins: number;
  features: Hist[];
}

export function FeatureHistograms({ data }: { data: HistogramsData | { missing_columns: string[] } }) {
  if ("missing_columns" in data) {
    return (
      <div className="rounded border border-amber-500/30 bg-amber-500/5 p-3 text-[11px] text-amber-200">
        Cannot render — missing <span className="font-mono">{data.missing_columns.join(", ")}</span>.
      </div>
    );
  }
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {data.features.map((f) => (
        <SingleHistogram key={f.column} feature={f} />
      ))}
    </div>
  );
}

function SingleHistogram({ feature }: { feature: Hist }) {
  const midpoints = feature.edges.slice(0, -1).map((e, i) => (e + feature.edges[i + 1]) / 2);
  const labels = midpoints.map((m) => m.toFixed(2));
  const option = {
    grid: { left: 36, right: 6, top: 18, bottom: 22 },
    tooltip: {
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    xAxis: {
      type: "category",
      data: labels,
      axisLabel: { color: "#64748b", fontSize: 8, interval: Math.floor(labels.length / 5) },
      axisTick: { show: false },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#64748b", fontSize: 8 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        type: "bar",
        data: feature.counts,
        itemStyle: { color: "#60a5fa" },
        barWidth: "95%",
      },
    ],
  } as const;
  return (
    <div className="rounded border border-slate-800 bg-slate-900/40 p-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-[11px] text-slate-200">{feature.column}</span>
        <span className="text-[10px] text-slate-500">
          μ={feature.mean ?? "–"} · σ={feature.std ?? "–"} · n={feature.n}
        </span>
      </div>
      <EChart option={option} height={140} data-chart="feature-histogram" />
    </div>
  );
}
