import type { ColumnUnit, SelectedDim, SelectedMeasure } from "./types";

export interface TidyResult {
  columns: string[];
  rows: (string | number | null)[][];
  n: number;
}

/** Group measures by unit into up to 3 axis-clusters. */
export function axisClusters(
  measures: SelectedMeasure[],
  columnUnit: (col: string) => ColumnUnit | undefined,
): { unit: ColumnUnit; measures: SelectedMeasure[] }[] {
  const groups = new Map<ColumnUnit, SelectedMeasure[]>();
  for (const m of measures) {
    const unit = m.axisOverride ?? columnUnit(m.column) ?? "count";
    if (!groups.has(unit)) groups.set(unit, []);
    groups.get(unit)!.push(m);
  }
  return Array.from(groups.entries())
    .slice(0, 3)
    .map(([unit, ms]) => ({ unit, measures: ms }));
}

/** Pivot a tidy result by the second dimension, returning a wide series array. */
export function pivotByGroup(
  data: TidyResult,
  xDim: SelectedDim,
  groupDim: SelectedDim,
  measure: SelectedMeasure,
): { xLabels: string[]; series: { name: string; data: number[] }[] } {
  const xAlias = xDim.alias ?? xDim.column;
  const gAlias = groupDim.alias ?? groupDim.column;
  const mAlias = measure.alias ?? `${measure.agg}_${measure.column}`;
  const xIdx = data.columns.indexOf(xAlias);
  const gIdx = data.columns.indexOf(gAlias);
  const mIdx = data.columns.indexOf(mAlias);
  if (xIdx < 0 || gIdx < 0 || mIdx < 0) return { xLabels: [], series: [] };

  const xSet = new Set<string>();
  const gSet = new Set<string>();
  for (const row of data.rows) {
    xSet.add(String(row[xIdx]));
    gSet.add(String(row[gIdx]));
  }
  const xLabels = Array.from(xSet).sort();
  const gLabels = Array.from(gSet).sort();
  const cellMap = new Map<string, number>();
  for (const row of data.rows) {
    cellMap.set(`${String(row[xIdx])}${String(row[gIdx])}`, Number(row[mIdx] ?? 0));
  }
  const series = gLabels.map((g) => ({
    name: g,
    data: xLabels.map((x) => cellMap.get(`${x}${g}`) ?? 0),
  }));
  return { xLabels, series };
}

/** Pluck the column for a given dim/measure alias. */
export function columnForAlias(data: TidyResult, alias: string): number {
  return data.columns.indexOf(alias);
}

/** Compute simple histogram bins for a single measure column. */
export function histogramBins(
  values: number[],
  bins = 20,
): { edges: number[]; counts: number[] } {
  if (!values.length) return { edges: [], counts: [] };
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return { edges: [min, max], counts: [values.length] };
  const step = (max - min) / bins;
  const edges = Array.from({ length: bins + 1 }, (_, i) => min + i * step);
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    let idx = Math.floor((v - min) / step);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx] += 1;
  }
  return { edges, counts };
}

export function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Per-category box stats (min, q1, median, q3, max). */
export function boxStats(
  data: TidyResult,
  dimAlias: string,
  measureAlias: string,
): { category: string; min: number; q1: number; median: number; q3: number; max: number }[] {
  const dimIdx = columnForAlias(data, dimAlias);
  const mIdx = columnForAlias(data, measureAlias);
  if (dimIdx < 0 || mIdx < 0) return [];
  const buckets = new Map<string, number[]>();
  for (const row of data.rows) {
    const key = String(row[dimIdx]);
    const val = Number(row[mIdx]);
    if (Number.isNaN(val)) continue;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(val);
  }
  return Array.from(buckets.entries()).map(([category, vs]) => {
    const sorted = [...vs].sort((a, b) => a - b);
    return {
      category,
      min: sorted[0],
      q1: quantile(sorted, 0.25),
      median: quantile(sorted, 0.5),
      q3: quantile(sorted, 0.75),
      max: sorted[sorted.length - 1],
    };
  });
}
