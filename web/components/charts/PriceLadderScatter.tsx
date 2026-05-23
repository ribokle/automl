"use client";

import { EChart } from "./EChart";

interface LadderPoint {
  price: number;
  n_weeks: number;
}

export interface PriceLadderData {
  ppg_id: string;
  ladder: LadderPoint[];
  price_volume_slope: number | null;
  log_intercept: number | null;
  caveat?: string;
}

export function PriceLadderScatter({ data }: { data: PriceLadderData }) {
  if (!data.ladder.length) {
    return <p className="text-[11px] text-slate-500">No price points for {data.ppg_id}.</p>;
  }
  const scatter = data.ladder.map((p) => [p.price, p.n_weeks]);
  const option = {
    tooltip: {
      trigger: "item",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      formatter: (p: { data: [number, number] }) =>
        `price $${p.data[0].toFixed(2)} · ${p.data[1]} week${p.data[1] === 1 ? "" : "s"}`,
    },
    grid: { left: 50, right: 12, top: 16, bottom: 32 },
    xAxis: {
      type: "value",
      name: "price ($)",
      nameLocation: "middle",
      nameGap: 22,
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "value",
      name: "# weeks at price",
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        type: "bar",
        data: scatter,
        itemStyle: { color: "#60a5fa" },
        barWidth: "70%",
      },
    ],
  } as const;
  return <EChart option={option} height={240} data-chart="price-ladder" />;
}
