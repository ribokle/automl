"use client";

import { memo } from "react";
import { EChart } from "./EChart";

export interface SimCell {
  price_multiplier: number;
  price: number;
  promo: number;
  units: number;
  revenue: number;
  margin: number;
}

export interface SimulationGridBlob {
  ppg_id: string;
  model_kind: string;
  cells: SimCell[];
}

type Metric = "revenue" | "margin" | "units";

interface Props {
  data: SimulationGridBlob;
  metric?: Metric;
  height?: number;
}

export const SimulationHeatmap = memo(function SimulationHeatmap({ data, metric = "revenue", height = 220 }: Props) {
  const cells = data.cells ?? [];
  if (!cells.length) {
    return <p className="text-[11px] text-slate-500">No simulation cells.</p>;
  }
  const multipliers = Array.from(new Set(cells.map((c) => c.price_multiplier))).sort(
    (a, b) => a - b,
  );
  const promos = Array.from(new Set(cells.map((c) => c.promo))).sort((a, b) => a - b);
  const promoLabel = (p: number) => (p === 1 ? "promo on" : "promo off");

  const points: [number, number, number][] = [];
  let lo = Infinity;
  let hi = -Infinity;
  for (const c of cells) {
    const x = multipliers.indexOf(c.price_multiplier);
    const y = promos.indexOf(c.promo);
    const v = c[metric];
    if (x < 0 || y < 0) continue;
    points.push([x, y, v]);
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }

  const option = {
    grid: { left: 80, right: 60, top: 24, bottom: 50 },
    tooltip: {
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      formatter: (p: { value: [number, number, number] }) => {
        const m = multipliers[p.value[0]];
        const pr = promos[p.value[1]];
        return `mult ${m.toFixed(2)}<br/>${promoLabel(pr)}<br/>${metric}: ${p.value[2].toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
      },
    },
    xAxis: {
      type: "category",
      name: "price multiplier",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      nameLocation: "middle",
      nameGap: 26,
      data: multipliers.map((m) => m.toFixed(2)),
      axisLabel: { color: "#64748b", fontSize: 9 },
    },
    yAxis: {
      type: "category",
      data: promos.map(promoLabel),
      axisLabel: { color: "#64748b", fontSize: 10 },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    visualMap: {
      min: Number.isFinite(lo) ? lo : 0,
      max: Number.isFinite(hi) ? hi : 1,
      calculable: true,
      orient: "vertical",
      right: 0,
      top: "middle",
      text: ["hi", "lo"],
      textStyle: { color: "#94a3b8", fontSize: 9 },
      inRange: { color: ["#1e293b", "#0ea5e9", "#fbbf24", "#34d399"] },
    },
    series: [
      {
        type: "heatmap",
        data: points,
        label: {
          show: true,
          color: "#0f172a",
          fontSize: 9,
          formatter: (p: { value: [number, number, number] }) => {
            const v = p.value[2];
            if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
            if (v >= 1_000) return `${(v / 1_000).toFixed(0)}k`;
            return v.toFixed(0);
          },
        },
        itemStyle: { borderColor: "#0f172a", borderWidth: 1 },
        emphasis: { itemStyle: { borderColor: "#fff", borderWidth: 2 } },
      },
    ],
  } as const;

  return <EChart option={option} height={height} data-chart="simulation-heatmap" />;
});
