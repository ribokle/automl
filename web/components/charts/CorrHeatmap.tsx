"use client";

import { EChart } from "./EChart";

export interface CorrData {
  labels: string[];
  matrix: number[][];
}

export function CorrHeatmap({ data, height }: { data: CorrData; height?: number }) {
  const cells: [number, number, number][] = [];
  for (let i = 0; i < data.labels.length; i++) {
    for (let j = 0; j < data.labels.length; j++) {
      cells.push([j, i, data.matrix[i][j]]);
    }
  }
  const option = {
    grid: { left: 100, right: 12, top: 90, bottom: 8 },
    tooltip: {
      formatter: (p: { value: [number, number, number] }) =>
        `${data.labels[p.value[1]]} × ${data.labels[p.value[0]]}<br/>r = ${p.value[2].toFixed(3)}`,
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    xAxis: {
      type: "category",
      data: data.labels,
      position: "top",
      axisLabel: { color: "#94a3b8", fontSize: 9, rotate: -50, align: "left" },
      axisTick: { show: false },
      axisLine: { show: false },
      splitArea: { show: false },
    },
    yAxis: {
      type: "category",
      data: data.labels,
      inverse: true,
      axisLabel: { color: "#94a3b8", fontSize: 9 },
      axisTick: { show: false },
      axisLine: { show: false },
    },
    visualMap: {
      show: false,
      min: -1,
      max: 1,
      inRange: { color: ["#f472b6", "#0f172a", "#34d399"] },
    },
    series: [
      {
        type: "heatmap",
        data: cells,
        label: {
          show: data.labels.length <= 14,
          formatter: (p: { value: [number, number, number] }) => (Math.abs(p.value[2]) >= 0.3 ? p.value[2].toFixed(2) : ""),
          fontSize: 8,
          color: "#0f172a",
        },
        itemStyle: { borderColor: "#0b1220", borderWidth: 0.5 },
      },
    ],
  } as const;
  return <EChart option={option} height={height ?? Math.max(260, data.labels.length * 22 + 110)} data-chart="corr-heatmap" />;
}
