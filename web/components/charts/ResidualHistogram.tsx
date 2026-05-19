"use client";

import { EChart } from "./EChart";

export interface ResidualRow {
  ppg_id: string;
  winner_model?: string;
  verdict?: string;
  residuals_log: number[];
  folds: { fold: number; n: number }[];
}

interface Props {
  data: ResidualRow;
  bins?: number;
  height?: number;
}

function histogram(values: number[], bins: number) {
  if (!values.length) return { edges: [0, 0], counts: [0] };
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const step = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    let idx = Math.floor((v - lo) / step);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx] += 1;
  }
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + i * step);
  return { edges, counts };
}

function mean(arr: number[]): number {
  if (!arr.length) return NaN;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stddev(arr: number[]): number {
  const n = arr.length;
  if (n < 2) return NaN;
  const m = mean(arr);
  let s = 0;
  for (const v of arr) s += (v - m) ** 2;
  return Math.sqrt(s / (n - 1));
}

export function ResidualHistogram({ data, bins = 18, height = 200 }: Props) {
  if (!data.residuals_log?.length) {
    return <p className="text-[11px] text-slate-500">No residuals.</p>;
  }
  const { edges, counts } = histogram(data.residuals_log, bins);
  const centers = counts.map((_, i) => ((edges[i] + edges[i + 1]) / 2).toFixed(3));
  const m = mean(data.residuals_log);
  const s = stddev(data.residuals_log);

  const option = {
    grid: { left: 50, right: 16, top: 28, bottom: 36 },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    xAxis: {
      type: "category",
      data: centers,
      name: "residual (log units)",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      nameLocation: "middle",
      nameGap: 24,
      axisLabel: { color: "#64748b", fontSize: 9, rotate: 35 },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    yAxis: {
      type: "value",
      name: "count",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        type: "bar",
        data: counts,
        itemStyle: { color: "#34d399", opacity: 0.85 },
        barCategoryGap: "10%",
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: "#fbbf24", type: "dashed", width: 1 },
          data: [{ name: "zero", xAxis: bins / 2 - 0.5 }],
        },
      },
    ],
    title: {
      text: `μ=${Number.isFinite(m) ? m.toFixed(3) : "—"} · σ=${Number.isFinite(s) ? s.toFixed(3) : "—"} · n=${data.residuals_log.length}`,
      textStyle: { color: "#94a3b8", fontSize: 10, fontWeight: "normal" },
      right: 8,
      top: 4,
    },
  } as const;

  return <EChart option={option} height={height} data-chart="residual-histogram" />;
}
