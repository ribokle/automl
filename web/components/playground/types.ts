export type ColumnRole = "dimension" | "measure" | "time";
export type ColumnUnit =
  | "count"
  | "dollars"
  | "share"
  | "ratio"
  | "category"
  | "date"
  | "id";

export interface ColumnMeta {
  role: ColumnRole;
  unit: ColumnUnit;
  label: string;
  default_agg: string;
}

export interface SchemaResponse {
  columns: Record<string, ColumnMeta>;
  dimensions: string[];
  measures: string[];
  time_columns: string[];
  aggregations: string[];
}

export interface SelectedDim {
  column: string;
  alias?: string;
}

export interface SelectedMeasure {
  column: string;
  agg: string;
  alias?: string;
  /** Optional unit override to force a particular y-axis assignment for trend charts. */
  axisOverride?: ColumnUnit;
}

export type RangeFilterValue = { gte?: string | number; lte?: string | number };

export interface FilterMap {
  [column: string]: string[] | RangeFilterValue;
}

export interface QuerySpec {
  dimensions: SelectedDim[];
  measures: SelectedMeasure[];
  filters: FilterMap;
  order_by?: { column: string; asc: boolean }[];
  limit?: number;
}
