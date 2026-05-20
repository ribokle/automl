"use client";

import { memo } from "react";

import { CHART_DEFAULTS } from "@/lib/chart-config";
import { EChart } from "./EChart";

export interface FittedVsActualRow {
  ppg_id: string;
  model: string;
  weeks: string[];
  observed_log: number[];
  predicted_log: number[];
  observed_units: number[];
  predicted_units: number[];
  split: ("train" | "test")[];
  n_train: number;
  n_test: number;
}

function corr(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n < 2) return NaN;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < n; i++) {
    sa += a[i];
    sb += b[i];
  }
  const ma = sa / n;
  const mb = sb / n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  const denom = Math.sqrt(va * vb);
  return denom > 0 ? cov / denom : NaN;
}

interface Props {
  data: FittedVsActualRow;
  height?: number;
}

export const FittedVsActual = memo(function FittedVsActual({
  data,
  height = CHART_DEFAULTS.height,
}: Props) {
  if (!data.observed_units.length) {
    return <p className="text-[11px] text-slate-500">No fitted-vs-actual data.</p>;
  }

  const trainPoints: [number, number][] = [];
  const testPoints: [number, number][] = [];
  for (let i = 0; i < data.observed_units.length; i++) {
    const point: [number, number] = [data.observed_units[i], data.predicted_units[i]];
    if (data.split[i] === "test") testPoints.push(point);
    else trainPoints.push(point);
  }

  const all = data.observed_units.concat(data.predicted_units);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = 0.05 * Math.max(1, hi - lo);
  const lineLo = Math.max(0, lo - pad);
  const lineHi = hi + pad;

  const r = corr(data.observed_log, data.predicted_log);

  const option = {
    grid: CHART_DEFAULTS.grid,
    legend: {
      data: ["train", "test", "y = x"],
      textStyle: { color: CHART_DEFAULTS.textColor, fontSize: CHART_DEFAULTS.fontSize },
      top: 4,
    },
    tooltip: {
      trigger: "item",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      formatter: (p: { seriesName: string; value: number[] }) =>
        `<b>${p.seriesName}</b><br/>observed: ${p.value[0].toFixed(0)}<br/>predicted: ${p.value[1].toFixed(0)}`,
    },
    xAxis: {
      type: "value",
      name: "observed units",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      nameLocation: "middle",
      nameGap: 26,
      min: lineLo,
      max: lineHi,
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "value",
      name: "predicted units",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      nameGap: 30,
      min: lineLo,
      max: lineHi,
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        name: "y = x",
        type: "line",
        showSymbol: false,
        lineStyle: { color: "#475569", type: "dashed", width: 1 },
        data: [
          [lineLo, lineLo],
          [lineHi, lineHi],
        ],
        z: 1,
      },
      {
        name: "train",
        type: "scatter",
        symbolSize: 6,
        itemStyle: { color: "#34d399", opacity: 0.55 },
        data: trainPoints,
        z: 2,
      },
      {
        name: "test",
        type: "scatter",
        symbolSize: 8,
        itemStyle: { color: "#fbbf24", opacity: 0.9 },
        data: testPoints,
        z: 3,
      },
    ],
    title: {
      text: Number.isFinite(r) ? `r (log) = ${r.toFixed(3)}` : "",
      textStyle: { color: "#94a3b8", fontSize: 10, fontWeight: "normal" },
      right: 8,
      top: 4,
    },
  } as const;

  return <EChart option={option} height={height} data-chart="fitted-vs-actual" />;
});
