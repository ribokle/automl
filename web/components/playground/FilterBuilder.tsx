"use client";

import { useEffect, useState } from "react";
import { distinctValues } from "@/lib/api";
import type { ColumnMeta, FilterMap, RangeFilterValue } from "./types";

export function FilterBuilder({
  runId,
  columns,
  dimensions,
  filters,
  onChange,
}: {
  runId: string;
  columns: Record<string, ColumnMeta>;
  dimensions: string[];
  filters: FilterMap;
  onChange: (next: FilterMap) => void;
}) {
  const [open, setOpen] = useState(false);
  const [pickColumn, setPickColumn] = useState<string>("");
  const [distinctMap, setDistinctMap] = useState<Record<string, (string | number)[]>>({});

  useEffect(() => {
    if (!pickColumn || distinctMap[pickColumn]) return;
    let cancelled = false;
    distinctValues(runId, pickColumn)
      .then((vs) => {
        if (!cancelled) setDistinctMap((m) => ({ ...m, [pickColumn]: vs }));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [pickColumn, runId, distinctMap]);

  const removeFilter = (col: string) => {
    const next = { ...filters };
    delete next[col];
    onChange(next);
  };
  const updateList = (col: string, vals: string[]) => {
    onChange({ ...filters, [col]: vals });
  };
  const updateRange = (col: string, range: RangeFilterValue) => {
    onChange({ ...filters, [col]: range });
  };

  const active = Object.keys(filters);
  return (
    <div className="space-y-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-slate-300 hover:text-slate-100"
      >
        <span>{open ? "▾" : "▸"}</span>
        Filters
        {active.length > 0 && (
          <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 text-[9px] text-emerald-200">
            {active.length}
          </span>
        )}
      </button>
      {open && (
        <div className="space-y-2 rounded border border-slate-800 bg-slate-900/40 p-2">
          <div className="flex gap-1">
            <select
              value={pickColumn}
              onChange={(e) => setPickColumn(e.target.value)}
              className="flex-1 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-[10px] text-slate-200"
            >
              <option value="">add filter…</option>
              {dimensions
                .filter((d) => !filters[d])
                .map((d) => (
                  <option key={d} value={d}>
                    {columns[d]?.label ?? d}
                  </option>
                ))}
            </select>
            <button
              type="button"
              disabled={!pickColumn}
              onClick={() => {
                if (!pickColumn) return;
                onChange({
                  ...filters,
                  [pickColumn]: columns[pickColumn]?.unit === "date" ? {} : [],
                });
                setPickColumn("");
              }}
              className="rounded border border-emerald-500/40 bg-emerald-500/10 px-2 py-1 text-[10px] text-emerald-200 hover:bg-emerald-500/20 disabled:opacity-40"
            >
              add
            </button>
          </div>
          {active.map((col) => {
            const value = filters[col];
            const isRange = !Array.isArray(value);
            return (
              <div key={col} className="rounded border border-slate-800 bg-slate-950 p-2">
                <div className="mb-1 flex items-center justify-between">
                  <span className="font-mono text-[11px] text-slate-200">
                    {columns[col]?.label ?? col}
                  </span>
                  <button
                    type="button"
                    onClick={() => removeFilter(col)}
                    className="text-[10px] text-slate-400 hover:text-rose-300"
                  >
                    remove
                  </button>
                </div>
                {isRange ? (
                  <div className="flex gap-1">
                    <input
                      type="text"
                      placeholder="gte (e.g. 2023-01-01)"
                      value={String((value as RangeFilterValue).gte ?? "")}
                      onChange={(e) =>
                        updateRange(col, {
                          ...(value as RangeFilterValue),
                          gte: e.target.value || undefined,
                        })
                      }
                      className="flex-1 rounded border border-slate-700 bg-slate-900 px-1 py-0.5 text-[10px] text-slate-200"
                    />
                    <input
                      type="text"
                      placeholder="lte"
                      value={String((value as RangeFilterValue).lte ?? "")}
                      onChange={(e) =>
                        updateRange(col, {
                          ...(value as RangeFilterValue),
                          lte: e.target.value || undefined,
                        })
                      }
                      className="flex-1 rounded border border-slate-700 bg-slate-900 px-1 py-0.5 text-[10px] text-slate-200"
                    />
                  </div>
                ) : (
                  <div className="flex max-h-32 flex-wrap gap-1 overflow-y-auto">
                    {(distinctMap[col] ?? []).map((v) => {
                      const sv = String(v);
                      const checked = (value as string[]).includes(sv);
                      return (
                        <label
                          key={sv}
                          className={`cursor-pointer rounded border px-2 py-0.5 text-[10px] ${
                            checked
                              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                              : "border-slate-700 bg-slate-900/60 text-slate-300 hover:bg-slate-800"
                          }`}
                        >
                          <input
                            type="checkbox"
                            className="mr-1 hidden"
                            checked={checked}
                            onChange={() => {
                              const cur = value as string[];
                              if (checked) {
                                updateList(
                                  col,
                                  cur.filter((x) => x !== sv),
                                );
                              } else {
                                updateList(col, [...cur, sv]);
                              }
                            }}
                          />
                          {sv}
                        </label>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
