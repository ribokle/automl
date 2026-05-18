"use client";

import { EChart } from "./EChart";

export interface PriceBoxData {
  colours: Record<string, string>;
  boxes: {
    ppg_id: string;
    min: number;
    q1: number;
    median: number;
    q3: number;
    max: number;
    n: number;
  }[];
}

export function PPGPriceBox({ data }: { data: PriceBoxData | { missing_columns: string[] } }) {
  if ("missing_columns" in data) {
    return (
      <div className="rounded border border-amber-500/30 bg-amber-500/5 p-3 text-[11px] text-amber-200">
        Cannot render — missing <span className="font-mono">{data.missing_columns.join(", ")}</span>.
      </div>
    );
  }
  const categories = data.boxes.map((b) => b.ppg_id);
  const boxData = data.boxes.map((b) => [b.min, b.q1, b.median, b.q3, b.max]);
  const itemColors = data.boxes.map((b) => data.colours[b.ppg_id] ?? "#94a3b8");
  const option = {
    grid: { left: 50, right: 12, top: 16, bottom: 50 },
    tooltip: {
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    xAxis: {
      type: "category",
      data: categories,
      axisLabel: { color: "#64748b", fontSize: 9, rotate: 35 },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    yAxis: {
      type: "value",
      name: "price ($)",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        type: "boxplot",
        data: boxData.map((d, i) => ({ value: d, itemStyle: { color: itemColors[i] + "44", borderColor: itemColors[i] } })),
      },
    ],
  } as const;
  return <EChart option={option} height={240} data-chart="ppg-price-box" />;
}
