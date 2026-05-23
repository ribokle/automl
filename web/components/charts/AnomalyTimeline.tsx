"use client";

import { EChart } from "./EChart";

interface AnomalyPoint {
  week_start: string;
  anomaly_type: string;
  severity: number;
  note?: string;
}

export interface AnomalyTimelineData {
  weeks: string[];
  by_ppg: Record<string, AnomalyPoint[]>;
}

const TYPE_COLOURS: Record<string, string> = {
  stockout: "#f43f5e",
  pantry_loading: "#a78bfa",
  forward_buy: "#fbbf24",
  isolation_forest: "#34d399",
};

export function AnomalyTimeline({ data, ppg }: { data: AnomalyTimelineData; ppg: string }) {
  const points = data.by_ppg[ppg] ?? [];
  if (points.length === 0) {
    return <p className="text-[11px] text-slate-500">No anomalies flagged for {ppg}.</p>;
  }
  const types = Array.from(new Set(points.map((p) => p.anomaly_type)));
  const series = types.map((t) => ({
    name: t,
    type: "scatter" as const,
    data: points
      .filter((p) => p.anomaly_type === t)
      .map((p) => ({
        value: [p.week_start, p.severity],
        tooltip: p.note ?? "",
      })),
    symbolSize: (v: [string, number]) => Math.max(8, Math.min(28, Math.abs(v[1]) * 6)),
    itemStyle: { color: TYPE_COLOURS[t] ?? "#60a5fa" },
  }));
  const option = {
    tooltip: {
      trigger: "item",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      formatter: (p: { data: { value: [string, number]; tooltip?: string }; seriesName: string }) =>
        `<b>${p.seriesName}</b><br/>${p.data.value[0]}<br/>severity ${p.data.value[1].toFixed(2)}<br/>${p.data.tooltip ?? ""}`,
    },
    legend: { data: types, textStyle: { color: "#cbd5e1", fontSize: 10 }, top: 4 },
    grid: { left: 50, right: 12, top: 32, bottom: 28 },
    xAxis: {
      type: "category",
      data: data.weeks,
      axisLabel: { color: "#64748b", fontSize: 9, interval: Math.max(1, Math.floor(data.weeks.length / 8)) },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    yAxis: {
      type: "value",
      name: "severity",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series,
  } as const;
  return <EChart option={option} height={260} data-chart="anomaly-timeline" />;
}
