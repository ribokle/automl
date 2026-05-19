"use client";

import { Fragment, useMemo, useState, type ReactNode } from "react";

export type Severity = "pass" | "warn" | "fail" | "info" | "neutral";

export interface ColumnDef<T> {
  /** Unique key for this column (also the default sort key). */
  key: string;
  /** Header text. */
  label: string;
  /** Cell alignment. Numeric columns default to right. */
  align?: "left" | "right" | "center";
  /** Custom cell renderer. Defaults to row[key]. */
  format?: (row: T) => ReactNode;
  /** Value used for sorting. Defaults to row[key]. */
  sortValue?: (row: T) => string | number | null | undefined;
  /** Disable sorting on this column. */
  unsortable?: boolean;
  /** Tabular-numeric / mono font for the cell. */
  numeric?: boolean;
  /** Severity for a header tooltip / column pill (optional). */
  severity?: Severity;
}

export interface ResultsTableProps<T> {
  rows: T[];
  columns: ColumnDef<T>[];
  rowKey: (row: T) => string;
  /** Show this when `rows` is empty. */
  empty?: string;
  /** Default sort. If absent, rows render in input order. */
  defaultSort?: { key: string; dir: "asc" | "desc" };
  /** Highlight a row by key (e.g. selected PPG). */
  highlightKey?: string | null;
  /** Click handler — receives the row. */
  onRowClick?: (row: T) => void;
  /** Optional row-level severity for left-edge accent (e.g. validation verdict). */
  rowSeverity?: (row: T) => Severity | null;
  /** Sticky-first-column shadow. Default true. */
  stickyFirst?: boolean;
  /** Compact density. Default true. */
  compact?: boolean;
  /** Expandable rows: adds a leading ▸/▾ column; expansion content
   *  rendered in a row below the parent row. */
  expandable?: {
    render: (row: T) => ReactNode;
    /** Initial expanded keys (one-time). */
    initialExpanded?: string[];
  };
}

const ROW_SEVERITY: Record<Severity, string> = {
  pass: "before:bg-emerald-400/70",
  warn: "before:bg-amber-400/70",
  fail: "before:bg-rose-500/70",
  info: "before:bg-sky-400/70",
  neutral: "before:bg-slate-700/70",
};

function compare(a: unknown, b: unknown): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

export function ResultsTable<T>({
  rows,
  columns,
  rowKey,
  empty = "No rows.",
  defaultSort,
  highlightKey,
  onRowClick,
  rowSeverity,
  stickyFirst = true,
  compact = true,
  expandable,
}: ResultsTableProps<T>) {
  const [sort, setSort] = useState<{ key: string; dir: "asc" | "desc" } | null>(
    defaultSort ?? null,
  );
  const [expanded, setExpanded] = useState<Set<string>>(
    () => new Set(expandable?.initialExpanded ?? []),
  );
  function toggleExpand(key: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col) return rows;
    const sv = col.sortValue ?? ((r: T) => (r as Record<string, unknown>)[col.key] as unknown);
    const list = [...rows].sort((a, b) => compare(sv(a), sv(b)));
    return sort.dir === "asc" ? list : list.reverse();
  }, [rows, sort, columns]);

  function toggleSort(key: string) {
    setSort((curr) => {
      if (!curr || curr.key !== key) return { key, dir: "desc" };
      if (curr.dir === "desc") return { key, dir: "asc" };
      return null;
    });
  }

  if (rows.length === 0) {
    return <p className="text-[11px] text-slate-500">{empty}</p>;
  }

  const padding = compact ? "px-2 py-1.5" : "px-3 py-2";
  const totalCols = columns.length + (expandable ? 1 : 0);

  return (
    <div className="overflow-x-auto rounded border border-slate-800">
      <table className="min-w-full text-[11px]">
        <thead className="bg-slate-900/80 text-[10px] uppercase tracking-wider text-slate-500">
          <tr>
            {expandable && <th className="w-7" aria-label="expand" />}
            {columns.map((c, i) => {
              const isSorted = sort?.key === c.key;
              const align = c.align ?? (c.numeric ? "right" : "left");
              const alignClass =
                align === "right" ? "text-right" : align === "center" ? "text-center" : "text-left";
              const stickyClass =
                i === 0 && stickyFirst
                  ? "sticky left-0 z-10 bg-slate-900/95 backdrop-blur"
                  : "";
              return (
                <th
                  key={c.key}
                  scope="col"
                  className={`${padding} ${alignClass} ${stickyClass} ${c.unsortable ? "" : "cursor-pointer select-none hover:text-slate-300"}`}
                  onClick={() => !c.unsortable && toggleSort(c.key)}
                  aria-sort={
                    isSorted ? (sort?.dir === "asc" ? "ascending" : "descending") : "none"
                  }
                >
                  <span className="inline-flex items-center gap-1">
                    {c.label}
                    {!c.unsortable && (
                      <span className={`text-[8px] ${isSorted ? "text-slate-300" : "text-slate-700"}`}>
                        {isSorted ? (sort?.dir === "asc" ? "▲" : "▼") : "↕"}
                      </span>
                    )}
                  </span>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/70">
          {sorted.map((row) => {
            const key = rowKey(row);
            const severity = rowSeverity?.(row);
            const highlight = highlightKey === key;
            const baseRow = onRowClick ? "cursor-pointer" : "";
            const sevAccent = severity
              ? `relative before:absolute before:left-0 before:top-0 before:h-full before:w-0.5 ${ROW_SEVERITY[severity]}`
              : "";
            const isExpanded = expandable ? expanded.has(key) : false;
            return (
              <Fragment key={key}>
                <tr
                  onClick={onRowClick ? () => onRowClick(row) : undefined}
                  className={`${baseRow} ${sevAccent} ${
                    highlight ? "bg-emerald-500/5" : "hover:bg-slate-900/40"
                  }`}
                >
                  {expandable && (
                    <td className="px-1 text-center align-middle">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleExpand(key);
                        }}
                        className="text-slate-500 hover:text-slate-300"
                        aria-label={isExpanded ? "collapse row" : "expand row"}
                        aria-expanded={isExpanded}
                      >
                        {isExpanded ? "▾" : "▸"}
                      </button>
                    </td>
                  )}
                  {columns.map((c, i) => {
                    const align = c.align ?? (c.numeric ? "right" : "left");
                    const alignClass =
                      align === "right"
                        ? "text-right"
                        : align === "center"
                          ? "text-center"
                          : "text-left";
                    const monoClass = c.numeric ? "font-mono tabular-nums" : "";
                    const stickyClass =
                      i === 0 && stickyFirst
                        ? "sticky left-0 z-0 bg-slate-950/80 backdrop-blur"
                        : "";
                    const value = c.format
                      ? c.format(row)
                      : String((row as Record<string, unknown>)[c.key] ?? "");
                    return (
                      <td
                        key={c.key}
                        className={`${padding} ${alignClass} ${monoClass} ${stickyClass} text-slate-200`}
                      >
                        {value}
                      </td>
                    );
                  })}
                </tr>
                {isExpanded && expandable && (
                  <tr className="bg-slate-950/40">
                    <td colSpan={totalCols} className="px-3 py-2">
                      {expandable.render(row)}
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
