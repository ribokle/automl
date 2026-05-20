"use client";

import { memo } from "react";

import { CHART_DEFAULTS } from "@/lib/chart-config";
import { EChart } from "./EChart";

export interface DecompWeekRow {
  week_start: string;
  observed: number;
  predicted: number;
  base: number;
  residual: number;
  due_by_group: Record<string, number>;
}

export interface DecompPPGBlob {
  ppg_id: string;
  weekly: DecompWeekRow[];
}

const GROUP_ORDER = [
  "base",
  "price",
  "promo",
  "distribution",
  "competitor",
  "seasonality",
  "lags",
  "other",
] as const;

const GROUP_COLOR: Record<string, string> = {
  base: "#475569",
  price: "#34d399",
  promo: "#fbbf24",
  distribution: "#a78bfa",
  competitor: "#f87171",
  seasonality: "#22d3ee",
  lags: "#fb923c",
  other: "#94a3b8",
};

interface Props {
  data: DecompPPGBlob;
  height?: number;
}

export const DecompStackedArea = memo(function DecompStackedArea({
  data,
  height = CHART_DEFAULTS.height,
}: Props) {
  const weekly = data.weekly ?? [];
  if (!weekly.length) {
    return <p className="text-[11px] text-slate-500">No decomposition data.</p>;
  }
  const weeks = weekly.map((w) => w.week_start);
  const observed = weekly.map((w) => w.observed);

  const presentGroups = new Set<string>(["base"]);
  for (const w of weekly) {
    for (const k of Object.keys(w.due_by_group ?? {})) {
      if (Math.abs(w.due_by_group[k]) > 1e-9) presentGroups.add(k);
    }
  }
  const groups = GROUP_ORDER.filter((g) => presentGroups.has(g));

  const series: Record<string, unknown>[] = groups.map((g) => ({
    name: g,
    type: "line",
    stack: "due",
    showSymbol: false,
    smooth: false,
    lineStyle: { width: 0 },
    areaStyle: { color: GROUP_COLOR[g] ?? "#94a3b8", opacity: 0.8 },
    emphasis: { focus: "series" },
    data: weekly.map((w) =>
      g === "base" ? w.base : (w.due_by_group?.[g] ?? 0),
    ),
  }));
  series.push({
    name: "observed",
    type: "line",
    showSymbol: false,
    smooth: true,
    lineStyle: { width: 1.5, color: "#e2e8f0" },
    emphasis: { focus: "series" },
    data: observed,
  });

  const option = {
    grid: { left: 60, right: 24, top: 32, bottom: 30 },
    legend: {
      data: [...groups, "observed"],
      textStyle: { color: "#cbd5e1", fontSize: 10 },
      top: 0,
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    xAxis: {
      type: "category",
      data: weeks,
      axisLabel: { color: "#64748b", fontSize: 9 },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    yAxis: {
      type: "value",
      name: "units",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series,
  } as const;

  return <EChart option={option} height={height} data-chart="decomp-stacked-area" />;
});
