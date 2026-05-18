"use client";

import { EChart } from "./EChart";

export interface VIFData {
  vif: Record<string, number>;
  threshold?: number;
}

export function VIFBar({ data }: { data: VIFData }) {
  const sorted = Object.entries(data.vif).sort((a, b) => b[1] - a[1]);
  const labels = sorted.map(([k]) => k);
  const values = sorted.map(([, v]) => Math.min(v, 15));
  const threshold = data.threshold ?? 10;
  const option = {
    grid: { left: 110, right: 28, top: 8, bottom: 30 },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    xAxis: {
      type: "value",
      max: Math.max(threshold + 1, Math.ceil(Math.max(...values))),
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "category",
      data: labels,
      inverse: true,
      axisLabel: { color: "#cbd5e1", fontSize: 10 },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    series: [
      {
        type: "bar",
        data: values.map((v) => ({
          value: v,
          itemStyle: { color: v >= threshold ? "#f43f5e" : v >= threshold * 0.7 ? "#fbbf24" : "#34d399" },
        })),
        barWidth: "55%",
        markLine: {
          symbol: "none",
          lineStyle: { color: "#f43f5e", type: "dashed", width: 1 },
          data: [{ xAxis: threshold, label: { formatter: `VIF ${threshold}`, color: "#fda4af", fontSize: 9 } }],
        },
      },
    ],
  } as const;
  return <EChart option={option} height={Math.max(220, labels.length * 24 + 50)} data-chart="vif-bar" />;
}
