"use client";

import { EChart } from "../charts/EChart";
import type { ChartTypeId } from "./chart_types";
import { axisClusters, boxStats, columnForAlias, histogramBins, pivotByGroup } from "./helpers";
import type { TidyResult } from "./helpers";
import type { ColumnMeta, SelectedDim, SelectedMeasure } from "./types";

const SERIES_COLOURS = [
  "#34d399", "#fbbf24", "#60a5fa", "#f472b6", "#a78bfa",
  "#22d3ee", "#fb7185", "#facc15", "#4ade80", "#38bdf8",
];

function colourAt(i: number): string {
  return SERIES_COLOURS[i % SERIES_COLOURS.length];
}

export function ChartRenderer({
  chartType,
  data,
  dims,
  measures,
  columns,
}: {
  chartType: ChartTypeId;
  data: TidyResult;
  dims: SelectedDim[];
  measures: SelectedMeasure[];
  columns: Record<string, ColumnMeta>;
}) {
  if (data.n === 0) {
    return <p className="text-[11px] text-slate-500">No rows for the current spec.</p>;
  }
  switch (chartType) {
    case "trend":
      return <TrendRenderer data={data} dims={dims} measures={measures} columns={columns} />;
    case "bar":
      return <BarRenderer data={data} dim={dims[0]} measure={measures[0]} />;
    case "grouped_bar":
    case "stacked_bar":
      return (
        <GroupedBarRenderer
          data={data}
          dim={dims[0]}
          groupDim={dims[1]}
          measure={measures[0]}
          stack={chartType === "stacked_bar"}
        />
      );
    case "scatter":
      return (
        <ScatterRenderer
          data={data}
          xMeasure={measures[0]}
          yMeasure={measures[1]}
          colourDim={dims[0]}
        />
      );
    case "heatmap":
      return (
        <HeatmapRenderer
          data={data}
          rowDim={dims[0]}
          colDim={dims[1]}
          measure={measures[0]}
        />
      );
    case "box":
      return <BoxRenderer data={data} dim={dims[0]} measure={measures[0]} />;
    case "histogram":
      return <HistogramRenderer data={data} measure={measures[0]} />;
    case "pareto":
      return <ParetoRenderer data={data} dim={dims[0]} measure={measures[0]} />;
    default:
      return null;
  }
}

function TrendRenderer({
  data,
  dims,
  measures,
  columns,
}: {
  data: TidyResult;
  dims: SelectedDim[];
  measures: SelectedMeasure[];
  columns: Record<string, ColumnMeta>;
}) {
  const xDim = dims[0];
  const xIdx = columnForAlias(data, xDim.alias ?? xDim.column);
  const xLabels = Array.from(new Set(data.rows.map((r) => String(r[xIdx])))).sort();

  const clusters = axisClusters(measures, (c) => columns[c]?.unit);
  const yAxis = clusters.map((c, i) => ({
    type: "value" as const,
    position: (i === 0 ? "left" : "right") as "left" | "right",
    offset: i > 1 ? 60 : 0,
    name: `${c.unit}${i > 0 ? ` (${c.measures.length})` : ""}`,
    nameTextStyle: { color: "#64748b", fontSize: 9 },
    axisLabel: { color: "#64748b", fontSize: 9 },
    splitLine: { lineStyle: { color: "#1e293b" } },
  }));

  let seriesIdx = 0;
  const groupDim = dims[1];
  const series = clusters.flatMap((cluster, clusterIdx) =>
    cluster.measures.flatMap((m) => {
      const mAlias = m.alias ?? `${m.agg}_${m.column}`;
      if (groupDim) {
        const pivoted = pivotByGroup(data, xDim, groupDim, m);
        return pivoted.series.map((s, gi) => {
          const colour = colourAt(seriesIdx++);
          return {
            name: `${columns[m.column]?.label ?? m.column} · ${s.name}`,
            type: "line" as const,
            yAxisIndex: clusterIdx,
            data: s.data,
            smooth: true,
            symbol: "none" as const,
            lineStyle: { color: colour, width: 2 },
            itemStyle: { color: colour },
          };
        });
      }
      const mIdx = columnForAlias(data, mAlias);
      const valByX = new Map<string, number>();
      for (const row of data.rows) {
        valByX.set(String(row[xIdx]), Number(row[mIdx] ?? 0));
      }
      const colour = colourAt(seriesIdx++);
      return [
        {
          name: columns[m.column]?.label ?? m.column,
          type: "line" as const,
          yAxisIndex: clusterIdx,
          data: xLabels.map((x) => valByX.get(x) ?? null),
          smooth: true,
          symbol: "none" as const,
          lineStyle: { color: colour, width: 2 },
          itemStyle: { color: colour },
        },
      ];
    }),
  );

  const option = {
    tooltip: {
      trigger: "axis",
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
    },
    legend: { textStyle: { color: "#cbd5e1", fontSize: 10 }, top: 4 },
    grid: { left: 60, right: clusters.length > 1 ? 70 : 20, top: 36, bottom: 32 },
    xAxis: {
      type: "category" as const,
      data: xLabels,
      axisLabel: {
        color: "#64748b",
        fontSize: 9,
        interval: Math.max(0, Math.floor(xLabels.length / 12)),
      },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    yAxis,
    series,
  };
  return <EChart option={option} height={320} data-chart="playground-trend" />;
}

function BarRenderer({
  data,
  dim,
  measure,
}: {
  data: TidyResult;
  dim: SelectedDim;
  measure: SelectedMeasure;
}) {
  const dimAlias = dim.alias ?? dim.column;
  const mAlias = measure.alias ?? `${measure.agg}_${measure.column}`;
  const dimIdx = columnForAlias(data, dimAlias);
  const mIdx = columnForAlias(data, mAlias);
  const pairs: [string, number][] = data.rows.map((r) => [String(r[dimIdx]), Number(r[mIdx] ?? 0)]);
  pairs.sort((a, b) => b[1] - a[1]);
  const option = {
    tooltip: { trigger: "axis", backgroundColor: "#0f172a", borderColor: "#334155", textStyle: { color: "#e2e8f0", fontSize: 11 } },
    grid: { left: 60, right: 20, top: 24, bottom: 60 },
    xAxis: {
      type: "category" as const,
      data: pairs.map((p) => p[0]),
      axisLabel: { color: "#64748b", fontSize: 9, rotate: -32 },
    },
    yAxis: {
      type: "value" as const,
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        type: "bar" as const,
        data: pairs.map((p) => p[1]),
        itemStyle: { color: "#34d399" },
        barMaxWidth: 32,
      },
    ],
  };
  return <EChart option={option} height={320} data-chart="playground-bar" />;
}

function GroupedBarRenderer({
  data,
  dim,
  groupDim,
  measure,
  stack,
}: {
  data: TidyResult;
  dim: SelectedDim;
  groupDim: SelectedDim;
  measure: SelectedMeasure;
  stack: boolean;
}) {
  const { xLabels, series } = pivotByGroup(data, dim, groupDim, measure);
  const option = {
    tooltip: { trigger: "axis", backgroundColor: "#0f172a", borderColor: "#334155", textStyle: { color: "#e2e8f0", fontSize: 11 } },
    legend: { textStyle: { color: "#cbd5e1", fontSize: 10 }, top: 4 },
    grid: { left: 60, right: 20, top: 32, bottom: 60 },
    xAxis: {
      type: "category" as const,
      data: xLabels,
      axisLabel: { color: "#64748b", fontSize: 9, rotate: -32 },
    },
    yAxis: {
      type: "value" as const,
      axisLabel: { color: "#64748b", fontSize: 9 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: series.map((s, i) => ({
      name: s.name,
      type: "bar" as const,
      data: s.data,
      ...(stack ? { stack: "total" } : {}),
      itemStyle: { color: colourAt(i) },
    })),
  };
  return <EChart option={option} height={320} data-chart="playground-grouped-bar" />;
}

function ScatterRenderer({
  data,
  xMeasure,
  yMeasure,
  colourDim,
}: {
  data: TidyResult;
  xMeasure: SelectedMeasure;
  yMeasure: SelectedMeasure;
  colourDim?: SelectedDim;
}) {
  const xIdx = columnForAlias(data, xMeasure.alias ?? `${xMeasure.agg}_${xMeasure.column}`);
  const yIdx = columnForAlias(data, yMeasure.alias ?? `${yMeasure.agg}_${yMeasure.column}`);
  const cIdx = colourDim ? columnForAlias(data, colourDim.alias ?? colourDim.column) : -1;
  if (cIdx >= 0) {
    const groups = new Map<string, [number, number, string][]>();
    for (const row of data.rows) {
      const key = String(row[cIdx]);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push([Number(row[xIdx]), Number(row[yIdx]), key]);
    }
    const series = Array.from(groups.entries()).map(([name, points], i) => ({
      name,
      type: "scatter" as const,
      data: points,
      itemStyle: { color: colourAt(i) },
      symbolSize: 8,
    }));
    const option = {
      tooltip: { trigger: "item", backgroundColor: "#0f172a", borderColor: "#334155", textStyle: { color: "#e2e8f0", fontSize: 11 } },
      legend: { textStyle: { color: "#cbd5e1", fontSize: 10 }, top: 4 },
      grid: { left: 60, right: 20, top: 32, bottom: 32 },
      xAxis: { type: "value" as const, name: xMeasure.column, axisLabel: { color: "#64748b", fontSize: 9 } },
      yAxis: { type: "value" as const, name: yMeasure.column, axisLabel: { color: "#64748b", fontSize: 9 } },
      series,
    };
    return <EChart option={option} height={320} data-chart="playground-scatter" />;
  }
  const points = data.rows.map((r) => [Number(r[xIdx]), Number(r[yIdx])]);
  const option = {
    tooltip: { trigger: "item", backgroundColor: "#0f172a", borderColor: "#334155", textStyle: { color: "#e2e8f0", fontSize: 11 } },
    grid: { left: 60, right: 20, top: 24, bottom: 32 },
    xAxis: { type: "value" as const, name: xMeasure.column, axisLabel: { color: "#64748b", fontSize: 9 } },
    yAxis: { type: "value" as const, name: yMeasure.column, axisLabel: { color: "#64748b", fontSize: 9 } },
    series: [{ type: "scatter" as const, data: points, itemStyle: { color: "#34d399" }, symbolSize: 6 }],
  };
  return <EChart option={option} height={320} data-chart="playground-scatter" />;
}

function HeatmapRenderer({
  data,
  rowDim,
  colDim,
  measure,
}: {
  data: TidyResult;
  rowDim: SelectedDim;
  colDim: SelectedDim;
  measure: SelectedMeasure;
}) {
  const rIdx = columnForAlias(data, rowDim.alias ?? rowDim.column);
  const cIdx = columnForAlias(data, colDim.alias ?? colDim.column);
  const mIdx = columnForAlias(data, measure.alias ?? `${measure.agg}_${measure.column}`);
  const rows = Array.from(new Set(data.rows.map((r) => String(r[rIdx])))).sort();
  const cols = Array.from(new Set(data.rows.map((r) => String(r[cIdx])))).sort();
  const rIndex = new Map(rows.map((r, i) => [r, i]));
  const cIndex = new Map(cols.map((c, i) => [c, i]));
  const cells: [number, number, number][] = data.rows.map((row) => [
    cIndex.get(String(row[cIdx])) ?? 0,
    rIndex.get(String(row[rIdx])) ?? 0,
    Number(row[mIdx] ?? 0),
  ]);
  const values = cells.map((c) => c[2]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const option = {
    tooltip: {
      backgroundColor: "#0f172a",
      borderColor: "#334155",
      textStyle: { color: "#e2e8f0", fontSize: 11 },
      formatter: (p: { value: [number, number, number] }) =>
        `${rows[p.value[1]]} · ${cols[p.value[0]]}<br/>${p.value[2].toFixed(2)}`,
    },
    grid: { left: 100, right: 60, top: 16, bottom: 36 },
    xAxis: {
      type: "category" as const,
      data: cols,
      axisLabel: { color: "#94a3b8", fontSize: 9, rotate: -32 },
    },
    yAxis: { type: "category" as const, data: rows, axisLabel: { color: "#94a3b8", fontSize: 9 } },
    visualMap: { show: true, min, max, inRange: { color: ["#0f172a", "#34d399", "#fbbf24"] }, textStyle: { color: "#94a3b8", fontSize: 9 } },
    series: [{ type: "heatmap" as const, data: cells, itemStyle: { borderColor: "#0b1220", borderWidth: 0.5 } }],
  };
  return <EChart option={option} height={Math.min(440, Math.max(220, rows.length * 22 + 60))} data-chart="playground-heatmap" />;
}

function BoxRenderer({
  data,
  dim,
  measure,
}: {
  data: TidyResult;
  dim: SelectedDim;
  measure: SelectedMeasure;
}) {
  const stats = boxStats(data, dim.alias ?? dim.column, measure.alias ?? `${measure.agg}_${measure.column}`);
  const cats = stats.map((s) => s.category);
  const boxes = stats.map((s) => [s.min, s.q1, s.median, s.q3, s.max]);
  const option = {
    tooltip: { trigger: "item", backgroundColor: "#0f172a", borderColor: "#334155", textStyle: { color: "#e2e8f0", fontSize: 11 } },
    grid: { left: 60, right: 20, top: 24, bottom: 60 },
    xAxis: { type: "category" as const, data: cats, axisLabel: { color: "#64748b", fontSize: 9, rotate: -32 } },
    yAxis: { type: "value" as const, axisLabel: { color: "#64748b", fontSize: 9 }, splitLine: { lineStyle: { color: "#1e293b" } } },
    series: [{ name: measure.column, type: "boxplot" as const, data: boxes, itemStyle: { color: "#60a5fa", borderColor: "#34d399" } }],
  };
  return <EChart option={option} height={320} data-chart="playground-box" />;
}

function HistogramRenderer({ data, measure }: { data: TidyResult; measure: SelectedMeasure }) {
  const mAlias = measure.alias ?? `${measure.agg}_${measure.column}`;
  const idx = columnForAlias(data, mAlias);
  const vals = data.rows.map((r) => Number(r[idx])).filter((v) => !Number.isNaN(v));
  const { edges, counts } = histogramBins(vals, 20);
  const labels = edges.slice(0, -1).map((e, i) => `${e.toFixed(2)}–${edges[i + 1].toFixed(2)}`);
  const option = {
    tooltip: { trigger: "axis", backgroundColor: "#0f172a", borderColor: "#334155", textStyle: { color: "#e2e8f0", fontSize: 11 } },
    grid: { left: 50, right: 20, top: 16, bottom: 60 },
    xAxis: { type: "category" as const, data: labels, axisLabel: { color: "#64748b", fontSize: 8, rotate: -32 } },
    yAxis: { type: "value" as const, axisLabel: { color: "#64748b", fontSize: 9 }, splitLine: { lineStyle: { color: "#1e293b" } } },
    series: [{ type: "bar" as const, data: counts, itemStyle: { color: "#34d399" }, barMaxWidth: 24 }],
  };
  return <EChart option={option} height={300} data-chart="playground-histogram" />;
}

function ParetoRenderer({
  data,
  dim,
  measure,
}: {
  data: TidyResult;
  dim: SelectedDim;
  measure: SelectedMeasure;
}) {
  const dimAlias = dim.alias ?? dim.column;
  const mAlias = measure.alias ?? `${measure.agg}_${measure.column}`;
  const dimIdx = columnForAlias(data, dimAlias);
  const mIdx = columnForAlias(data, mAlias);
  const pairs: [string, number][] = data.rows.map((r) => [String(r[dimIdx]), Number(r[mIdx] ?? 0)]);
  pairs.sort((a, b) => b[1] - a[1]);
  const total = pairs.reduce((s, p) => s + p[1], 0) || 1;
  let cum = 0;
  const cumPct: number[] = pairs.map((p) => {
    cum += p[1];
    return (cum / total) * 100;
  });
  const option = {
    tooltip: { trigger: "axis", backgroundColor: "#0f172a", borderColor: "#334155", textStyle: { color: "#e2e8f0", fontSize: 11 } },
    legend: { textStyle: { color: "#cbd5e1", fontSize: 10 }, top: 4 },
    grid: { left: 60, right: 60, top: 32, bottom: 60 },
    xAxis: {
      type: "category" as const,
      data: pairs.map((p) => p[0]),
      axisLabel: { color: "#64748b", fontSize: 9, rotate: -32 },
    },
    yAxis: [
      { type: "value" as const, name: measure.column, axisLabel: { color: "#64748b", fontSize: 9 }, splitLine: { lineStyle: { color: "#1e293b" } } },
      { type: "value" as const, name: "cum %", max: 100, axisLabel: { color: "#64748b", fontSize: 9, formatter: "{value}%" }, splitLine: { show: false } },
    ],
    series: [
      { name: measure.column, type: "bar" as const, data: pairs.map((p) => p[1]), itemStyle: { color: "#34d399" }, barMaxWidth: 32 },
      { name: "cumulative %", type: "line" as const, yAxisIndex: 1, data: cumPct.map((v) => v.toFixed(2)), itemStyle: { color: "#fbbf24" }, lineStyle: { color: "#fbbf24", width: 2 }, smooth: false, symbol: "circle", symbolSize: 4 },
    ],
  };
  return <EChart option={option} height={320} data-chart="playground-pareto" />;
}
