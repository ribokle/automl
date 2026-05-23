"use client";

import { EChart } from "./EChart";

export interface ACFData {
  available: boolean;
  ppg_id?: string;
  lags?: number[];
  acf?: number[];
  pacf?: number[];
  ci?: number;
  reason?: string;
}

export function ACFPlot({ data }: { data: ACFData }) {
  if (!data.available || !data.lags) {
    return (
      <p className="text-[11px] text-slate-500">
        ACF unavailable: {data.reason ?? "insufficient data"}
      </p>
    );
  }
  const lags = data.lags;
  const ci = data.ci ?? 0;
  const option = {
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    legend: { data: ["ACF", "PACF"], textStyle: { color: "#cbd5e1", fontSize: 10 }, top: 4 },
    grid: [
      { left: 50, right: 12, top: 32, height: "40%" },
      { left: 50, right: 12, top: "55%", height: "40%" },
    ],
    xAxis: [0, 1].map((i) => ({
      gridIndex: i,
      type: "category" as const,
      data: lags.map((l) => String(l)),
      axisLabel: { color: "#64748b", fontSize: 9 },
      axisLine: { lineStyle: { color: "#334155" } },
    })),
    yAxis: [
      { gridIndex: 0, name: "ACF", nameTextStyle: { color: "#64748b", fontSize: 10 }, axisLabel: { color: "#64748b", fontSize: 9 }, splitLine: { lineStyle: { color: "#1e293b" } } },
      { gridIndex: 1, name: "PACF", nameTextStyle: { color: "#64748b", fontSize: 10 }, axisLabel: { color: "#64748b", fontSize: 9 }, splitLine: { lineStyle: { color: "#1e293b" } } },
    ],
    series: [
      {
        name: "ACF",
        type: "bar",
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: data.acf,
        itemStyle: { color: "#34d399" },
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: "#f43f5e", type: "dashed", width: 1 },
          data: [{ yAxis: ci }, { yAxis: -ci }],
        },
      },
      {
        name: "PACF",
        type: "bar",
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: data.pacf,
        itemStyle: { color: "#fbbf24" },
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: "#f43f5e", type: "dashed", width: 1 },
          data: [{ yAxis: ci }, { yAxis: -ci }],
        },
      },
    ],
  } as const;
  return <EChart option={option} height={320} data-chart="acf" />;
}
