"use client";

import { EChart } from "./EChart";

export interface EligibilityData {
  threshold: number;
  bars: {
    ppg_id: string;
    score: number;
    eligible: boolean;
    contributions: {
      volume: number;
      coverage: number;
      price_variation: number;
      promo_variation: number;
    };
  }[];
}

const METRIC_COLOURS = {
  volume: "#34d399",
  coverage: "#60a5fa",
  price_variation: "#fbbf24",
  promo_variation: "#a78bfa",
};

export function EligibilityBars({ data }: { data: EligibilityData | { missing_columns: string[] } }) {
  if ("missing_columns" in data) {
    return (
      <div className="rounded border border-amber-500/30 bg-amber-500/5 p-3 text-[11px] text-amber-200">
        Cannot render — missing <span className="font-mono">{data.missing_columns.join(", ")}</span>.
      </div>
    );
  }
  const categories = data.bars.map((b) => b.ppg_id);
  const series = (Object.keys(METRIC_COLOURS) as (keyof typeof METRIC_COLOURS)[]).map((k) => ({
    name: k.replace("_", " "),
    type: "bar",
    stack: "score",
    data: data.bars.map((b) => b.contributions[k]),
    itemStyle: { color: METRIC_COLOURS[k] },
    barWidth: "55%",
  }));
  const option = {
    grid: { left: 90, right: 32, top: 24, bottom: 12 },
    legend: { top: 0, textStyle: { color: "#cbd5e1", fontSize: 10 } },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    xAxis: {
      type: "value",
      max: 1,
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "category",
      data: categories,
      axisLabel: { color: "#cbd5e1", fontSize: 10 },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    series: [
      ...series,
      {
        type: "line",
        markLine: {
          symbol: "none",
          lineStyle: { color: "#f43f5e", type: "dashed", width: 1 },
          data: [{ xAxis: data.threshold, label: { formatter: `eligible ≥ ${data.threshold}`, color: "#fda4af", fontSize: 9 } }],
        },
        data: [],
      },
    ],
  } as const;
  return <EChart option={option} height={Math.max(220, data.bars.length * 28 + 60)} data-chart="eligibility-bars" />;
}
