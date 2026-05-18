"use client";

import { EChart } from "./EChart";

export interface TrendData {
  weeks: string[];
  units: number[];
  price: number[];
  promo_share: number[];
}

export function TrendChart({ data }: { data: TrendData }) {
  const option = {
    grid: { left: 50, right: 50, top: 32, bottom: 32 },
    legend: {
      data: ["units", "avg price"],
      textStyle: { color: "#cbd5e1", fontSize: 11 },
      top: 4,
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    xAxis: {
      type: "category",
      data: data.weeks,
      axisLabel: { color: "#64748b", fontSize: 9 },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    yAxis: [
      {
        type: "value",
        name: "units",
        nameTextStyle: { color: "#64748b", fontSize: 10 },
        axisLabel: { color: "#64748b", fontSize: 9 },
        splitLine: { lineStyle: { color: "#1e293b" } },
      },
      {
        type: "value",
        name: "price",
        nameTextStyle: { color: "#64748b", fontSize: 10 },
        axisLabel: { color: "#64748b", fontSize: 9 },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: "units",
        type: "line",
        data: data.units,
        smooth: true,
        itemStyle: { color: "#34d399" },
        lineStyle: { width: 2 },
        symbol: "none",
      },
      {
        name: "avg price",
        type: "line",
        yAxisIndex: 1,
        data: data.price,
        smooth: true,
        itemStyle: { color: "#fbbf24" },
        lineStyle: { width: 2 },
        symbol: "none",
      },
    ],
  } as const;
  return <EChart option={option} height={240} data-chart="trend" />;
}
