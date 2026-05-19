"use client";

import { EChart } from "./EChart";

export interface SHAPSummary {
  method: "tree_shap" | "ols_centred";
  n_rows: number;
  n_features: number;
  base_value: number;
  mean_abs_shap: { feature: string; value: number }[];
  mean_shap: { feature: string; value: number }[];
}

export interface SHAPBarProps {
  data: SHAPSummary;
  height?: number;
}

export function SHAPBar({ data, height }: SHAPBarProps) {
  const rows = data.mean_abs_shap.slice(0, 10);
  const labels = rows.map((r) => r.feature);
  const meanAbs = rows.map((r) => r.value);
  const signedByFeature = new Map(data.mean_shap.map((r) => [r.feature, r.value]));
  const option = {
    grid: { left: 130, right: 28, top: 8, bottom: 30 },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      formatter: (params: { axisValue: string; value: number }[]) => {
        const f = params[0].axisValue;
        const signed = signedByFeature.get(f) ?? 0;
        return `${f}<br/>mean|SHAP| = ${params[0].value.toFixed(3)}<br/>mean SHAP = ${signed.toFixed(3)}`;
      },
    },
    xAxis: {
      type: "value",
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "category",
      data: labels,
      inverse: true,
      axisLabel: { color: "#cbd5e1", fontSize: 10 },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    series: [
      {
        type: "bar",
        data: rows.map((r) => ({
          value: r.value,
          itemStyle: {
            color: (signedByFeature.get(r.feature) ?? 0) < 0 ? "#60a5fa" : "#34d399",
          },
        })),
        barWidth: "55%",
      },
    ],
  } as const;
  return (
    <EChart
      option={option}
      height={height ?? Math.max(180, labels.length * 24 + 50)}
      data-chart="shap-bar"
    />
  );
}
