"use client";

import type { ChartTypeMeta } from "./chart_types";
import type { ColumnMeta, SelectedDim } from "./types";

export function DimensionPicker({
  available,
  columns,
  selected,
  onChange,
  meta,
}: {
  available: string[];
  columns: Record<string, ColumnMeta>;
  selected: SelectedDim[];
  onChange: (next: SelectedDim[]) => void;
  meta: ChartTypeMeta;
}) {
  const selectedSet = new Set(selected.map((s) => s.column));
  const canAdd = selected.length < meta.dims.max;

  const add = (col: string) => {
    if (!canAdd || selectedSet.has(col)) return;
    onChange([...selected, { column: col }]);
  };
  const remove = (col: string) => {
    onChange(selected.filter((s) => s.column !== col));
  };
  const move = (col: string, direction: -1 | 1) => {
    const idx = selected.findIndex((s) => s.column === col);
    if (idx < 0) return;
    const j = idx + direction;
    if (j < 0 || j >= selected.length) return;
    const next = [...selected];
    [next[idx], next[j]] = [next[j], next[idx]];
    onChange(next);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">Dimensions</h4>
        <span className={`text-[10px] ${selected.length < meta.dims.min ? "text-amber-300" : "text-slate-500"}`}>
          {selected.length} / {meta.dims.max} · need {meta.dims.min}+
        </span>
      </div>
      <p className="text-[10px] italic text-slate-500">{meta.dims.help}</p>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((s, i) => (
            <span
              key={s.column}
              className="inline-flex items-center gap-1 rounded border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-200"
            >
              <span className="font-mono">{columns[s.column]?.label ?? s.column}</span>
              {meta.ordered_dims && selected.length > 1 && (
                <>
                  <button
                    type="button"
                    onClick={() => move(s.column, -1)}
                    className="text-emerald-400 hover:text-emerald-200 disabled:opacity-30"
                    disabled={i === 0}
                    title="move left"
                  >
                    ◀
                  </button>
                  <button
                    type="button"
                    onClick={() => move(s.column, 1)}
                    className="text-emerald-400 hover:text-emerald-200 disabled:opacity-30"
                    disabled={i === selected.length - 1}
                    title="move right"
                  >
                    ▶
                  </button>
                </>
              )}
              <button
                type="button"
                onClick={() => remove(s.column)}
                className="text-emerald-400 hover:text-rose-300"
                title="remove"
              >
                ×
              </button>
            </span>
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
