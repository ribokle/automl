"use client";

import { EChart } from "./EChart";

export interface STLData {
  available: boolean;
  ppg_id?: string;
  weeks?: string[];
  observed?: number[];
  trend?: number[];
  seasonal?: number[];
  resid?: number[];
  seasonal_amplitude?: number | null;
  trend_range?: number | null;
  reason?: string;
}

export function STLDecomposition({ data }: { data: STLData }) {
  if (!data.available || !data.weeks) {
    return (
      <p className="text-[11px] text-slate-500">
        STL unavailable: {data.reason ?? "insufficient data"}
      </p>
    );
  }
  const weeks = data.weeks;
  const series = (name: string, values: number[], color: string, grid: number) => ({
    name,
    type: "line",
    smooth: true,
    symbol: "none",
    xAxisIndex: grid,
    yAxisIndex: grid,
    data: values,
    lineStyle: { width: 1.5, color },
    itemStyle: { color },
  });
  const option = {
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "cross" },
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    grid: [
      { left: 50, right: 12, top: 16, height: "18%" },
      { left: 50, right: 12, top: "32%", height: "18%" },
      { left: 50, right: 12, top: "55%", height: "18%" },
      { left: 50, right: 12, top: "78%", height: "18%" },
    ],
    xAxis: [0, 1, 2, 3].map((i) => ({
      gridIndex: i,
      type: "category" as const,
      data: weeks,
      axisLabel: { show: i === 3, color: "#64748b", fontSize: 8, interval: Math.max(1, Math.floor(weeks.length / 8)) },
      axisLine: { lineStyle: { color: "#334155" } },
    })),
    yAxis: [
      { gridIndex: 0, name: "observed", nameTextStyle: { color: "#64748b", fontSize: 9 }, axisLabel: { color: "#64748b", fontSize: 8 }, splitLine: { lineStyle: { color: "#1e293b" } } },
      { gridIndex: 1, name: "trend", nameTextStyle: { color: "#64748b", fontSize: 9 }, axisLabel: { color: "#64748b", fontSize: 8 }, splitLine: { lineStyle: { color: "#1e293b" } } },
      { gridIndex: 2, name: "seasonal", nameTextStyle: { color: "#64748b", fontSize: 9 }, axisLabel: { color: "#64748b", fontSize: 8 }, splitLine: { lineStyle: { color: "#1e293b" } } },
      { gridIndex: 3, name: "resid", nameTextStyle: { color: "#64748b", fontSize: 9 }, axisLabel: { color: "#64748b", fontSize: 8 }, splitLine: { lineStyle: { color: "#1e293b" } } },
    ],
    series: [
      series("observed", data.observed ?? [], "#cbd5e1", 0),
      series("trend", data.trend ?? [], "#fbbf24", 1),
      series("seasonal", data.seasonal ?? [], "#34d399", 2),
      series("resid", data.resid ?? [], "#f472b6", 3),
    ],
  } as const;
  return <EChart option={option} height={420} data-chart="stl" />;
}
