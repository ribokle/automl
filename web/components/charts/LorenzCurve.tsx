"use client";

import { EChart } from "./EChart";

interface LorenzSeries {
  x: number[];
  cum_share: number[];
  labels: string[];
}

export interface LorenzData {
  dimension: string;
  volume: LorenzSeries;
  revenue: LorenzSeries;
}

export function LorenzCurve({ data }: { data: LorenzData }) {
  const datasets: Array<[string, LorenzSeries, string]> = [
    ["units", data.volume, "#34d399"],
    ["revenue", data.revenue, "#fbbf24"],
  ];
  const series = datasets
    .filter(([, s]) => s.x.length > 0)
    .map(([name, s, color]) => ({
      name,
      type: "line" as const,
      smooth: false,
      symbol: "none",
      data: s.x.map((x, i) => [x, s.cum_share[i]]),
      lineStyle: { width: 2, color },
      areaStyle: { opacity: 0.08, color },
    }));
  const option = {
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      valueFormatter: (v: number) => `${(v * 100).toFixed(1)}%`,
    },
    legend: { data: series.map((s) => s.name), textStyle: { color: "#cbd5e1", fontSize: 10 }, top: 4 },
    grid: { left: 50, right: 12, top: 32, bottom: 32 },
    xAxis: {
      type: "value",
      name: `${data.dimension} share (sorted)`,
      nameLocation: "middle",
      nameGap: 22,
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      min: 0,
      max: 1,
      axisLabel: {
        color: "#64748b",
        fontSize: 9,
        formatter: (v: number) => `${(v * 100).toFixed(0)}%`,
      },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "value",
      name: "cumulative %",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      min: 0,
      max: 1,
      axisLabel: {
        color: "#64748b",
        fontSize: 9,
        formatter: (v: number) => `${(v * 100).toFixed(0)}%`,
      },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series,
  } as const;
  return <EChart option={option} height={260} data-chart={`lorenz-${data.dimension}`} />;
}
