export type ChartTypeId =
  | "trend"
  | "bar"
  | "grouped_bar"
  | "stacked_bar"
  | "scatter"
  | "heatmap"
  | "box"
  | "histogram"
  | "pareto";

export interface ChartTypeMeta {
  id: ChartTypeId;
  label: string;
  blurb: string;
  dims: { min: number; max: number; help: string };
  measures: { min: number; max: number; help: string };
  ordered_dims: boolean;
  multi_axis: boolean;
  prefer_time_first: boolean;
}

export const CHART_TYPES: Record<ChartTypeId, ChartTypeMeta> = {
  trend: {
    id: "trend",
    label: "Trend",
    blurb: "Line over time, one or more measures",
    dims: { min: 1, max: 2, help: "1 time dim (e.g. Week). Optional 2nd categorical dim to split lines." },
    measures: { min: 1, max: 5, help: "1-5 measures. Mixed scales auto-split onto separate y-axes." },
    ordered_dims: true,
    multi_axis: true,
    prefer_time_first: true,
  },
  bar: {
    id: "bar",
    label: "Bar",
    blurb: "One categorical dim, one measure",
    dims: { min: 1, max: 1, help: "1 categorical dim on the x-axis." },
    measures: { min: 1, max: 1, help: "Exactly 1 measure on the y-axis." },
    ordered_dims: true,
    multi_axis: false,
    prefer_time_first: false,
  },
  grouped_bar: {
    id: "grouped_bar",
    label: "Grouped bar",
    blurb: "Two dims · grouped side-by-side",
    dims: { min: 2, max: 2, help: "1st dim = x-axis category, 2nd dim = group within each x." },
    measures: { min: 1, max: 1, help: "Exactly 1 measure." },
    ordered_dims: true,
    multi_axis: false,
    prefer_time_first: false,
  },
  stacked_bar: {
    id: "stacked_bar",
    label: "Stacked bar",
    blurb: "Two dims · stacked",
    dims: { min: 2, max: 2, help: "1st dim = x-axis, 2nd dim = stack segment." },
    measures: { min: 1, max: 1, help: "Exactly 1 measure." },
    ordered_dims: true,
    multi_axis: false,
    prefer_time_first: false,
  },
  scatter: {
    id: "scatter",
    label: "Scatter",
    blurb: "Two measures, optional colour",
    dims: { min: 0, max: 1, help: "Optional 1 dim for point colour." },
    measures: { min: 2, max: 2, help: "Exactly 2 measures: x and y." },
    ordered_dims: true,
    multi_axis: false,
    prefer_time_first: false,
  },
  heatmap: {
    id: "heatmap",
    label: "Heatmap",
    blurb: "Two dims, one measure",
    dims: { min: 2, max: 2, help: "1st dim = rows, 2nd dim = columns." },
    measures: { min: 1, max: 1, help: "Exactly 1 measure for cell colour." },
    ordered_dims: true,
    multi_axis: false,
    prefer_time_first: false,
  },
  box: {
    id: "box",
    label: "Box",
    blurb: "Distribution per category",
    dims: { min: 1, max: 1, help: "1 categorical dim." },
    measures: { min: 1, max: 1, help: "1 measure (raw values are bucketed per dim)." },
    ordered_dims: true,
    multi_axis: false,
    prefer_time_first: false,
  },
  histogram: {
    id: "histogram",
    label: "Histogram",
    blurb: "Distribution of a single measure",
    dims: { min: 0, max: 0, help: "No dims." },
    measures: { min: 1, max: 1, help: "1 measure, binned into 20 buckets." },
    ordered_dims: false,
    multi_axis: false,
    prefer_time_first: false,
  },
  pareto: {
    id: "pareto",
    label: "Pareto",
    blurb: "Sorted bars + cumulative %",
    dims: { min: 1, max: 1, help: "1 categorical dim ranked by the measure." },
    measures: { min: 1, max: 1, help: "1 measure (auto sorted descending)." },
    ordered_dims: true,
    multi_axis: true,
    prefer_time_first: false,
  },
};

export const CHART_TYPE_LIST: ChartTypeMeta[] = Object.values(CHART_TYPES);
