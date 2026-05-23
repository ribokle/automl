"use client";

import { EChart } from "./EChart";

export interface PromoCalendarData {
  weeks: string[];
  ppgs: string[];
  matrix: string[][];
}

const TYPE_VALUE: Record<string, number> = {
  none: 0,
  tpr: 1,
  display: 2,
  feature: 3,
  multi: 4,
};

const TYPE_COLOURS = ["#0f172a", "#fbbf24", "#34d399", "#60a5fa", "#a78bfa"];

export function PromoCalendarHeatmap({ data }: { data: PromoCalendarData }) {
  if (!data.weeks.length || !data.ppgs.length) {
    return <p className="text-[11px] text-slate-500">No promo activity to display.</p>;
  }
  const cells: [number, number, number, string][] = [];
  for (let i = 0; i < data.ppgs.length; i++) {
    for (let j = 0; j < data.weeks.length; j++) {
      const cell = data.matrix[i][j];
      cells.push([j, i, TYPE_VALUE[cell] ?? 0, cell]);
    }
  }
  const option = {
    tooltip: {
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      formatter: (p: { value: [number, number, number, string] }) =>
        `${data.ppgs[p.value[1]]} · ${data.weeks[p.value[0]]}<br/>${p.value[3]}`,
    },
    grid: { left: 80, right: 12, top: 4, bottom: 24 },
    xAxis: {
      type: "category",
      data: data.weeks,
      axisLabel: { color: "#64748b", fontSize: 8, interval: Math.max(1, Math.floor(data.weeks.length / 12)) },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    yAxis: {
      type: "category",
      data: data.ppgs,
      axisLabel: { color: "#94a3b8", fontSize: 9 },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    visualMap: {
      show: false,
      type: "piecewise",
      pieces: [
        { value: 0, label: "none", color: TYPE_COLOURS[0] },
        { value: 1, label: "tpr", color: TYPE_COLOURS[1] },
        { value: 2, label: "display", color: TYPE_COLOURS[2] },
        { value: 3, label: "feature", color: TYPE_COLOURS[3] },
        { value: 4, label: "multi", color: TYPE_COLOURS[4] },
      ],
    },
    series: [
      {
        type: "heatmap",
        data: cells.map((c) => [c[0], c[1], c[2]]),
        progressive: 1000,
        itemStyle: { borderColor: "#0b1220", borderWidth: 0.5 },
      },
    ],
  } as const;
  return <EChart option={option} height={Math.min(420, Math.max(160, data.ppgs.length * 22 + 40))} data-chart="promo-calendar" />;
}
