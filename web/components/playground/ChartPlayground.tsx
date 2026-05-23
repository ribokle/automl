"use client";

import { useEffect, useMemo, useState } from "react";
import { getQuerySchema, runQuery } from "@/lib/api";
import { ChartRenderer } from "./ChartRenderer";
import { ChartTypePicker } from "./ChartTypePicker";
import { CHART_TYPES, type ChartTypeId } from "./chart_types";
import { DimensionPicker } from "./DimensionPicker";
import { FilterBuilder } from "./FilterBuilder";
import { MeasurePicker } from "./MeasurePicker";
import type { TidyResult } from "./helpers";
import type {
  FilterMap,
  QuerySpec,
  SchemaResponse,
  SelectedDim,
  SelectedMeasure,
} from "./types";

export function ChartPlayground({ runId }: { runId: string }) {
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [chartType, setChartType] = useState<ChartTypeId>("trend");
  const [dims, setDims] = useState<SelectedDim[]>([]);
  const [measures, setMeasures] = useState<SelectedMeasure[]>([]);
  const [filters, setFilters] = useState<FilterMap>({});
  const [data, setData] = useState<TidyResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getQuerySchema()
      .then((s) => setSchema(s as SchemaResponse))
      .catch((e) => setSchemaError(String(e)));
  }, []);

  // Reset to chart-type defaults when the user switches types.
  useEffect(() => {
    if (!schema) return;
    const meta = CHART_TYPES[chartType];
    setData(null);
    setError(null);
    // Trim to max dims/measures; if the user is starting fresh, give a sensible seed.
    setDims((prev) => {
      const trimmed = prev.slice(0, meta.dims.max);
      if (trimmed.length === 0 && meta.dims.min > 0 && meta.prefer_time_first && schema.time_columns.length > 0) {
        return [{ column: schema.time_columns[0] }];
      }
      return trimmed;
    });
    setMeasures((prev) => prev.slice(0, meta.measures.max));
  }, [chartType, schema]);

  const meta = CHART_TYPES[chartType];
  const ready =
    dims.length >= meta.dims.min &&
    dims.length <= meta.dims.max &&
    measures.length >= meta.measures.min &&
    measures.length <= meta.measures.max;

  const missingHint = useMemo(() => {
    const parts: string[] = [];
    if (dims.length < meta.dims.min) {
      parts.push(`need ${meta.dims.min - dims.length} more dim${meta.dims.min - dims.length === 1 ? "" : "s"}`);
    }
    if (measures.length < meta.measures.min) {
      parts.push(
        `need ${meta.measures.min - measures.length} more measure${meta.measures.min - measures.length === 1 ? "" : "s"}`,
      );
    }
    return parts.join(", ");
  }, [meta, dims.length, measures.length]);

  const runIt = async () => {
    if (!ready) return;
    setRunning(true);
    setError(null);
    const spec: QuerySpec = {
      dimensions: dims,
      measures,
      filters,
      limit: 5000,
    };
    if (meta.prefer_time_first && dims[0]) {
      spec.order_by = [{ column: dims[0].alias ?? dims[0].column, asc: true }];
    }
    try {
      const result = await runQuery<TidyResult>(runId, spec);
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setData(null);
    } finally {
      setRunning(false);
    }
  };

  if (schemaError) {
    return (
      <p className="text-[11px] text-rose-300">Failed to load schema: {schemaError}</p>
    );
  }
  if (!schema) {
    return <p className="text-[11px] text-slate-500">Loading schema…</p>;
  }

  return (
    <div className="space-y-4">
      <p className="text-[10.5px] italic text-slate-500">
        Build a chart from the panel mart. Pick a chart type, add dimensions
        and measures, optionally filter — the query runs against{" "}
        <code className="rounded bg-slate-800 px-1 font-mono text-[10px]">main.panel</code> and the
        result renders here.
      </p>
      <ChartTypePicker value={chartType} onChange={setChartType} />
      <div className="grid gap-4 md:grid-cols-2">
        <DimensionPicker
          available={schema.dimensions}
          columns={schema.columns}
          selected={dims}
          onChange={setDims}
          meta={meta}
        />
        <MeasurePicker
          available={schema.measures}
          columns={schema.columns}
          aggregations={schema.aggregations}
          selected={measures}
          onChange={setMeasures}
          meta={meta}
        />
      </div>
      <FilterBuilder
        runId={runId}
        columns={schema.columns}
        dimensions={schema.dimensions}
        filters={filters}
        onChange={setFilters}
      />
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={runIt}
          disabled={!ready || running}
          title={!ready ? `Render disabled — ${missingHint}` : "Run the query"}
          className="rounded border border-emerald-500/40 bg-emerald-500/15 px-4 py-1.5 text-[11px] font-semibold text-emerald-100 hover:bg-emerald-500/25 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {running ? "Running…" : "Render chart"}
        </button>
        {!ready && (
          <span className="text-[10px] text-amber-300">{missingHint}</span>
        )}
        {error && <span className="text-[10px] text-rose-300">Error: {error}</span>}
        {data && (
          <span className="text-[10px] text-slate-500">
            {data.n} row{data.n === 1 ? "" : "s"}
          </span>
        )}
      </div>
      {data && data.n > 0 && (
        <div className="rounded border border-slate-800 bg-slate-900/40 p-2">
          <ChartRenderer
            chartType={chartType}
            data={data}
            dims={dims}
            measures={measures}
            columns={schema.columns}
          />
        </div>
      )}
    </div>
  );
}
