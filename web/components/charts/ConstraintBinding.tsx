"use client";

import { memo } from "react";
import { EChart } from "./EChart";

export interface ConstraintBindingRow {
  ppg_id: string;
  slacks: Record<string, number>;
  feasible_strict: boolean;
  relaxed: boolean;
}

const CONSTRAINT_LABEL: Record<string, string> = {
  margin_floor: "margin floor",
  move_lower: "move ↓",
  move_upper: "move ↑",
  comp_gap_lower: "comp gap ↓",
  comp_gap_upper: "comp gap ↑",
};

interface Props {
  rows: ConstraintBindingRow[];
  height?: number;
}

/**
 * Plot, per PPG, how far each chosen cell sits from each constraint.
 *
 * Positive value = slack (constraint not binding). Negative value = the
 * constraint was violated (only happens when the MILP relaxed). Bars
 * close to zero indicate the cell is at the boundary — the constraint
 * is what stopped the optimiser from pushing further.
 */
export const ConstraintBinding = memo(function ConstraintBinding({ rows, height }: Props) {
  if (!rows.length) {
    return (
      <p className="text-[11px] text-slate-500">
        No constraint diagnostics available.
      </p>
    );
  }

  const keys = new Set<string>();
  for (const r of rows) {
    for (const k of Object.keys(r.slacks ?? {})) keys.add(k);
  }
  const constraints = Array.from(keys);
  const ppgs = rows.map((r) => r.ppg_id);

  const series = constraints.map((c) => ({
    name: CONSTRAINT_LABEL[c] ?? c,
    type: "bar",
    data: rows.map((r) => {
      const v = r.slacks?.[c];
      return v === undefined || !Number.isFinite(v) ? null : v;
    }),
    emphasis: { focus: "series" },
    itemStyle: {
      borderRadius: 2,
    },
  }));

  const option = {
    grid: { left: 100, right: 24, top: 32, bottom: 30 },
    legend: {
      data: constraints.map((c) => CONSTRAINT_LABEL[c] ?? c),
      textStyle: { color: "#cbd5e1", fontSize: 10 },
      top: 0,
    },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      valueFormatter: (v: number | null) => (v == null ? "—" : v.toFixed(3)),
    },
    xAxis: {
      type: "value",
      name: "slack (negative = violated)",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      nameLocation: "middle",
      nameGap: 24,
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "category",
      data: ppgs,
      axisLabel: { color: "#cbd5e1", fontSize: 10 },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    series,
  } as const;

  return (
    <EChart
      option={option}
      height={height ?? Math.max(160, ppgs.length * 32 + 60)}
      data-chart="constraint-binding"
    />
  );
});
