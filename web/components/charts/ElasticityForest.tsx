"use client";

import { EChart } from "./EChart";

export interface PosteriorRow {
  ppg_id: string;
  point: number;
  std_err: number;
  shrunk_mean: number;
  shrunk_std: number;
  ci_low: number;
  ci_high: number;
  shrinkage_weight: number;
}

export interface PosteriorBlob {
  population_mean: number;
  tau_squared: number;
  q_statistic: number;
  n_studies: number;
  per_ppg: PosteriorRow[];
}

const Z_95 = 1.96;

interface ForestProps {
  data: PosteriorBlob;
  height?: number;
}

export function ElasticityForest({ data, height }: ForestProps) {
  if (!data.per_ppg.length) {
    return <p className="text-[11px] text-slate-500">No OLS winners to pool.</p>;
  }
  // Sort by OLS point estimate so the most-elastic PPGs sit at the top.
  const sorted = [...data.per_ppg].sort((a, b) => a.point - b.point);
  const labels = sorted.map((r) => r.ppg_id);
  const rawPoints = sorted.map((r, i) => [r.point, i]);
  const rawErr = sorted.map((r, i) => [r.point - Z_95 * r.std_err, r.point + Z_95 * r.std_err, i]);
  const shrunkPoints = sorted.map((r, i) => [r.shrunk_mean, i]);
  const shrunkErr = sorted.map((r, i) => [r.ci_low, r.ci_high, i]);
  const minX = Math.min(
    ...rawErr.map((p) => p[0]),
    ...shrunkErr.map((p) => p[0]),
    0,
  );
  const maxX = Math.max(
    ...rawErr.map((p) => p[1]),
    ...shrunkErr.map((p) => p[1]),
    0,
  );
  const pad = 0.1 * Math.max(0.5, maxX - minX);

  const renderWhisker = (
    color: string,
  ) => ({
    type: "custom" as const,
    renderItem: (
      _params: unknown,
      api: { value: (i: number) => number; coord: (v: [number, number]) => [number, number]; size?: (v: [number, number]) => number[] },
    ) => {
      const yIdx = api.value(2);
      const low = api.coord([api.value(0), yIdx]);
      const high = api.coord([api.value(1), yIdx]);
      const tickHalf = 4;
      return {
        type: "group",
        children: [
          {
            type: "line",
            shape: { x1: low[0], y1: low[1], x2: high[0], y2: high[1] },
            style: { stroke: color, lineWidth: 1.5 },
          },
          {
            type: "line",
            shape: { x1: low[0], y1: low[1] - tickHalf, x2: low[0], y2: low[1] + tickHalf },
            style: { stroke: color, lineWidth: 1.5 },
          },
          {
            type: "line",
            shape: { x1: high[0], y1: high[1] - tickHalf, x2: high[0], y2: high[1] + tickHalf },
            style: { stroke: color, lineWidth: 1.5 },
          },
        ],
      };
    },
    encode: { x: [0, 1], y: 2 },
  });

  const option = {
    grid: { left: 130, right: 32, top: 28, bottom: 36 },
    legend: {
      data: ["OLS point ± 95% CI", "Shrunken posterior ± 95% CI"],
      textStyle: { color: "#cbd5e1", fontSize: 10 },
      top: 0,
    },
    tooltip: {
      trigger: "item",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      formatter: (params: { seriesName: string; value: number[]; dataIndex: number }) => {
        const row = sorted[params.dataIndex];
        if (params.seriesName.startsWith("OLS")) {
          const lo = (row.point - Z_95 * row.std_err).toFixed(2);
          const hi = (row.point + Z_95 * row.std_err).toFixed(2);
          return `<b>${row.ppg_id}</b><br/>OLS β = ${row.point.toFixed(2)}<br/>95% CI: [${lo}, ${hi}]<br/>SE = ${row.std_err.toFixed(3)}`;
        }
        return `<b>${row.ppg_id}</b><br/>Shrunken β̃ = ${row.shrunk_mean.toFixed(2)}<br/>95% CI: [${row.ci_low.toFixed(2)}, ${row.ci_high.toFixed(2)}]<br/>Shrinkage weight: ${(row.shrinkage_weight * 100).toFixed(0)}%`;
      },
    },
    xAxis: {
      type: "value",
      min: minX - pad,
      max: maxX + pad,
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "category",
      data: labels,
      axisLabel: { color: "#cbd5e1", fontSize: 10 },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    series: [
      {
        ...renderWhisker("#475569"),
        name: "OLS point ± 95% CI",
        data: rawErr,
        z: 1,
      },
      {
        name: "OLS point ± 95% CI",
        type: "scatter",
        symbolSize: 8,
        itemStyle: { color: "#94a3b8" },
        data: rawPoints,
        z: 2,
      },
      {
        ...renderWhisker("#34d399"),
        name: "Shrunken posterior ± 95% CI",
        data: shrunkErr,
        z: 3,
      },
      {
        name: "Shrunken posterior ± 95% CI",
        type: "scatter",
        symbolSize: 10,
        symbol: "diamond",
        itemStyle: { color: "#10b981" },
        data: shrunkPoints,
        z: 4,
      },
      {
        type: "line",
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: "#f43f5e", type: "dashed", width: 1 },
          data: [{ xAxis: 0, label: { formatter: "β=0", color: "#fda4af", fontSize: 9 } }],
        },
        data: [],
      },
      {
        type: "line",
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: "#fbbf24", type: "dotted", width: 1 },
          data: [
            {
              xAxis: data.population_mean,
              label: {
                formatter: `μ̂=${data.population_mean.toFixed(2)}`,
                color: "#fde68a",
                fontSize: 9,
              },
            },
          ],
        },
        data: [],
      },
    ],
  } as const;

  return (
    <EChart
      option={option}
      height={height ?? Math.max(220, labels.length * 28 + 70)}
      data-chart="elasticity-forest"
    />
  );
}
