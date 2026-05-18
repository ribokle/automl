"use client";

import { EChart } from "./EChart";

interface ScatterPoint {
  ppg_id: string;
  sku: string;
  x: number;
  y: number;
  pack?: string;
  brand?: string;
}

interface BaseScatter {
  view: "tier" | "behaviour" | "facet";
  x_label: string;
  y_label: string;
  colours: Record<string, string>;
}

export interface TierOrBehaviourData extends BaseScatter {
  points: ScatterPoint[];
}

export interface FacetData extends BaseScatter {
  facets: { category: string; brands: string[]; points: ScatterPoint[] }[];
}

export function PPGScatter({ data }: { data: TierOrBehaviourData | FacetData | { missing_columns: string[] } }) {
  if ("missing_columns" in data) {
    return <MissingColumns columns={data.missing_columns} />;
  }
  if (data.view === "facet") return <FacetScatter data={data as FacetData} />;
  return <SingleScatter data={data as TierOrBehaviourData} />;
}

function SingleScatter({ data }: { data: TierOrBehaviourData }) {
  const byPpg = new Map<string, ScatterPoint[]>();
  for (const p of data.points) {
    const list = byPpg.get(p.ppg_id) ?? [];
    list.push(p);
    byPpg.set(p.ppg_id, list);
  }
  const series = Array.from(byPpg.entries()).map(([ppg, pts]) => ({
    name: ppg,
    type: "scatter",
    symbolSize: 9,
    itemStyle: { color: data.colours[ppg] ?? "#94a3b8" },
    data: pts.map((p) => [p.x, p.y, p.sku, p.pack ?? p.brand ?? ""]),
  }));
  const option = {
    grid: { left: 50, right: 12, top: 30, bottom: 36 },
    legend: { top: 4, textStyle: { color: "#cbd5e1", fontSize: 10 } },
    tooltip: {
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      formatter: (p: { value: [number, number, string, string]; seriesName: string }) =>
        `${p.value[2]} · ${p.seriesName}${p.value[3] ? ` · ${p.value[3]}` : ""}<br/>${data.x_label}=${p.value[0].toFixed(2)}<br/>${data.y_label}=${p.value[1].toFixed(3)}`,
    },
    xAxis: {
      type: "value",
      name: data.x_label,
      nameLocation: "middle",
      nameGap: 22,
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "value",
      name: data.y_label,
      nameLocation: "middle",
      nameGap: 38,
      nameTextStyle: { color: "#64748b", fontSize: 10 },
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series,
  } as const;
  return <EChart option={option} height={300} data-chart="ppg-scatter" />;
}

function FacetScatter({ data }: { data: FacetData }) {
  const n = data.facets.length;
  const cols = Math.min(n, 3);
  const rows = Math.ceil(n / cols);
  const grid = data.facets.map((_, i) => {
    const r = Math.floor(i / cols);
    const c = i % cols;
    return {
      left: `${(c / cols) * 100 + 4}%`,
      width: `${100 / cols - 6}%`,
      top: `${(r / rows) * 100 + 14}%`,
      height: `${100 / rows - 22}%`,
    };
  });
  const xAxis = data.facets.map((f, i) => ({
    gridIndex: i,
    type: "value" as const,
    min: -0.5,
    max: Math.max(0, f.brands.length - 0.5),
    axisLabel: {
      color: "#64748b",
      fontSize: 8,
      formatter: (v: number) => f.brands[Math.round(v)] ?? "",
    },
    splitLine: { show: false },
  }));
  const yAxis = data.facets.map((_, i) => ({
    gridIndex: i,
    type: "value" as const,
    min: -0.5,
    max: 2.5,
    axisLabel: {
      color: "#64748b",
      fontSize: 8,
      formatter: (v: number) => ["S", "M", "L"][Math.round(v)] ?? "",
    },
    splitLine: { lineStyle: { color: "#1e293b" } },
  }));
  const titles = data.facets.map((f, i) => ({
    text: f.category,
    left: grid[i].left,
    top: `${Math.floor(i / cols) * (100 / rows) + 4}%`,
    textStyle: { color: "#cbd5e1", fontSize: 11, fontWeight: "normal" as const },
  }));
  const series = data.facets.flatMap((f, i) => {
    const byPpg = new Map<string, ScatterPoint[]>();
    for (const p of f.points) {
      const list = byPpg.get(p.ppg_id) ?? [];
      list.push(p);
      byPpg.set(p.ppg_id, list);
    }
    return Array.from(byPpg.entries()).map(([ppg, pts]) => ({
      name: ppg,
      type: "scatter" as const,
      xAxisIndex: i,
      yAxisIndex: i,
      symbolSize: 8,
      itemStyle: { color: data.colours[ppg] ?? "#94a3b8" },
      data: pts.map((p) => [p.x, p.y, p.sku, p.brand ?? "", p.pack ?? ""]),
    }));
  });
  const option = {
    title: titles,
    grid,
    xAxis,
    yAxis,
    tooltip: {
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      formatter: (p: { value: [number, number, string, string, string] }) =>
        `${p.value[2]} · ${p.value[3]} · ${p.value[4]}`,
    },
    series,
  } as const;
  const h = Math.max(260, rows * 180);
  return <EChart option={option} height={h} data-chart="ppg-scatter-facet" />;
}

function MissingColumns({ columns }: { columns: string[] }) {
  return (
    <div className="rounded border border-amber-500/30 bg-amber-500/5 p-3 text-[11px] text-amber-200">
      Cannot render — uploaded dataset is missing the column{columns.length > 1 ? "s" : ""}{" "}
      <span className="font-mono">{columns.join(", ")}</span>.
    </div>
  );
}
