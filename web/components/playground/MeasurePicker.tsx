"use client";

import type { ChartTypeMeta } from "./chart_types";
import type { ColumnMeta, ColumnUnit, SelectedMeasure } from "./types";

const AXIS_UNITS: ColumnUnit[] = ["count", "dollars", "share", "ratio"];

export function MeasurePicker({
  available,
  columns,
  aggregations,
  selected,
  onChange,
  meta,
}: {
  available: string[];
  columns: Record<string, ColumnMeta>;
  aggregations: string[];
  selected: SelectedMeasure[];
  onChange: (next: SelectedMeasure[]) => void;
  meta: ChartTypeMeta;
}) {
  const selectedSet = new Set(selected.map((s) => s.column));
  const canAdd = selected.length < meta.measures.max;

  const add = (col: string) => {
    if (!canAdd || selectedSet.has(col)) return;
    const default_agg = columns[col]?.default_agg ?? "sum";
    onChange([...selected, { column: col, agg: default_agg }]);
  };
  const remove = (col: string) => {
    onChange(selected.filter((s) => s.column !== col));
  };
  const updateAgg = (col: string, agg: string) => {
    onChange(selected.map((s) => (s.column === col ? { ...s, agg } : s)));
  };
  const updateAxis = (col: string, axisOverride: ColumnUnit | undefined) => {
    onChange(selected.map((s) => (s.column === col ? { ...s, axisOverride } : s)));
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">Measures</h4>
        <span
          className={`text-[10px] ${selected.length < meta.measures.min ? "text-amber-300" : "text-slate-500"}`}
        >
          {selected.length} / {meta.measures.max} · need {meta.measures.min}+
        </span>
      </div>
      <p className="text-[10px] italic text-slate-500">{meta.measures.help}</p>
      {selected.length > 0 && (
        <div className="flex flex-col gap-1">
          {selected.map((s) => (
            <div
              key={s.column}
              className="flex items-center gap-2 rounded border border-emerald-500/30 bg-emerald-500/5 px-2 py-1 text-[11px]"
            >
              <span className="flex-1 font-mono text-emerald-200">
                {columns[s.column]?.label ?? s.column}
              </span>
              <select
                value={s.agg}
                onChange={(e) => updateAgg(s.column, e.target.value)}
                className="rounded border border-slate-700 bg-slate-900 px-1 py-0.5 text-[10px] text-slate-200"
              >
                {aggregations.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
              {meta.multi_axis && (
                <select
                  value={s.axisOverride ?? ""}
                  onChange={(e) =>
                    updateAxis(s.column, (e.target.value || undefined) as ColumnUnit | undefined)
                  }
                  className="rounded border border-slate-700 bg-slate-900 px-1 py-0.5 text-[10px] text-slate-200"
                  title="Force this measure onto a specific axis"
                >
                  <option value="">axis: auto</option>
                  {AXIS_UNITS.map((u) => (
                    <option key={u} value={u}>
                      axis: {u}
                    </option>
                  ))}
                </select>
              )}
              <button
                type="button"
                onClick={() => remove(s.column)}
                className="text-emerald-400 hover:text-rose-300"
                title="remove"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="flex flex-wrap gap-1">
        {available
          .filter((c) => !selectedSet.has(c))
          .map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => add(c)}
              disabled={!canAdd}
              className="rounded border border-slate-700 bg-slate-900/60 px-2 py-0.5 text-[10px] text-slate-300 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              + {columns[c]?.label ?? c}
            </button>
          ))}
      </div>
    </div>
  );
}
