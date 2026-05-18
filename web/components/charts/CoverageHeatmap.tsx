"use client";

import { EChart } from "./EChart";

export interface CoverageData {
  skus: string[];
  weeks: string[];
  present: number[][];
  n_total_cells: number;
  n_present_cells: number;
}

export function CoverageHeatmap({ data }: { data: CoverageData }) {
  const cells: [number, number, number][] = [];
  for (let i = 0; i < data.skus.length; i++) {
    for (let j = 0; j < data.weeks.length; j++) {
      cells.push([j, i, data.present[i][j]]);
    }
  }
  const option = {
    grid: { left: 80, right: 12, top: 4, bottom: 20 },
    tooltip: {
      formatter: (p: { value: [number, number, number] }) =>
        `${data.skus[p.value[1]]} · ${data.weeks[p.value[0]]} · ${p.value[2] ? "present" : "missing"}`,
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    xAxis: {
      type: "category",
      data: data.weeks,
      show: false,
    },
    yAxis: {
      type: "category",
      data: data.skus,
      axisLabel: { color: "#64748b", fontSize: 8 },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    visualMap: {
      show: false,
      min: 0,
      max: 1,
      inRange: { color: ["#f43f5e", "#34d399"] },
    },
    series: [
      {
        type: "heatmap",
        data: cells,
        itemStyle: { borderColor: "#0b1220", borderWidth: 0.5 },
        progressive: 1000,
      },
    ],
  } as const;
  return (
    <EChart
      option={option}
      height={Math.min(420, Math.max(180, data.skus.length * 8 + 30))}
      data-chart="coverage-heatmap"
    />
  );
}
