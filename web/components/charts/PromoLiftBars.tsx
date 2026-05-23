"use client";

import { EChart } from "./EChart";

interface LiftEntry {
  lift: number;
  ci_lo: number;
  ci_hi: number;
  n_on: number;
}

export interface PromoLiftRow {
  ppg_id: string;
  lifts: {
    tpr?: LiftEntry;
    display?: LiftEntry;
    feature?: LiftEntry;
  };
}

export function PromoLiftBars({ rows }: { rows: PromoLiftRow[] }) {
  if (!rows.length) {
    return <p className="text-[11px] text-slate-500">No lift data available.</p>;
  }
  const ppgs = rows.map((r) => r.ppg_id);
  const types: Array<["tpr" | "display" | "feature", string]> = [
    ["tpr", "#fbbf24"],
    ["display", "#34d399"],
    ["feature", "#60a5fa"],
  ];
  const series = types.flatMap(([t, color]) => [
    {
      name: t,
      type: "bar" as const,
      data: rows.map((r) => r.lifts[t]?.lift ?? 0),
      itemStyle: { color },
      barGap: 0.1,
    },
    {
      name: `${t} ci`,
      type: "custom" as const,
      renderItem: (
        params: { dataIndex: number },
        api: {
          value: (i: number) => number;
          coord: (xy: [number, number]) => [number, number];
          size: (xy: [number, number]) => [number, number];
        },
      ) => {
        const idx = params.dataIndex;
        const row = rows[idx];
        const entry = row.lifts[t];
        if (!entry) return null;
        const offset = types.findIndex(([tt]) => tt === t) - 1;
        const x = api.coord([idx + offset * 0.3, 0])[0];
        const [yLo, yHi] = [api.coord([0, entry.ci_lo])[1], api.coord([0, entry.ci_hi])[1]];
        return {
          type: "line",
          shape: { x1: x, y1: yLo, x2: x, y2: yHi },
          style: { stroke: "#cbd5e1", lineWidth: 1 },
        };
      },
      data: rows.map((_, i) => i),
      legendHoverLink: false,
      silent: true,
      tooltip: { show: false },
    },
  ]);
  const option = {
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    legend: {
      data: types.map(([t]) => t),
      textStyle: { color: "#cbd5e1", fontSize: 10 },
      top: 4,
    },
    grid: { left: 50, right: 12, top: 32, bottom: 36 },
    xAxis: {
      type: "category",
      data: ppgs,
      axisLabel: { color: "#94a3b8", fontSize: 9, rotate: -28 },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    yAxis: {
      type: "value",
      name: "lift (×)",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    series,
  } as const;
  return <EChart option={option} height={Math.min(380, Math.max(220, ppgs.length * 28 + 80))} data-chart="promo-lift" />;
}
