"use client";

import { ResultsTable, type ColumnDef, type Severity } from "./ResultsTable";

export interface ValidationRow {
  ppg_id: string;
  winner: string;
  verdict: "pass" | "warn" | "fail";
  sign_stability: number;
  wape_mean: number;
  elasticity_mean: number;
  elasticity_cv: number;
  n_folds: number;
  rationale?: string;
}

const VERDICT_STYLE: Record<ValidationRow["verdict"], string> = {
  pass: "border-emerald-500/40 bg-emerald-500/15 text-emerald-300",
  warn: "border-amber-500/40 bg-amber-500/15 text-amber-300",
  fail: "border-rose-500/40 bg-rose-500/15 text-rose-300",
};

const VERDICT_ORDER: Record<ValidationRow["verdict"], number> = { fail: 0, warn: 1, pass: 2 };

function fmtPct(v: number, digits = 0): string {
  if (!Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

function fmt(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

const COLUMNS: ColumnDef<ValidationRow>[] = [
  { key: "ppg_id", label: "PPG" },
  {
    key: "verdict",
    label: "Verdict",
    align: "center",
    sortValue: (r) => VERDICT_ORDER[r.verdict] ?? 3,
    format: (r) => (
      <span
        className={`rounded border px-1.5 py-0.5 text-[9.5px] uppercase ${VERDICT_STYLE[r.verdict]}`}
      >
        {r.verdict}
      </span>
    ),
  },
  {
    key: "sign_stability",
    label: "Sign stab.",
    numeric: true,
    format: (r) => fmtPct(r.sign_stability),
  },
  { key: "wape_mean", label: "WAPE", numeric: true, format: (r) => fmt(r.wape_mean, 3) },
  {
    key: "elasticity_mean",
    label: "ε mean",
    numeric: true,
    format: (r) => fmt(r.elasticity_mean),
  },
  {
    key: "elasticity_cv",
    label: "ε CV",
    numeric: true,
    format: (r) => fmt(r.elasticity_cv),
  },
  { key: "n_folds", label: "Folds", numeric: true },
  {
    key: "winner",
    label: "Winner",
    format: (r) => <span className="text-slate-400">{r.winner}</span>,
  },
];

function severity(row: ValidationRow): Severity {
  return row.verdict;
}

export function ValidationTable({ rows }: { rows: ValidationRow[] }) {
  return (
    <ResultsTable
      rows={rows}
      columns={COLUMNS}
      rowKey={(r) => r.ppg_id}
      empty="No validation results."
      defaultSort={{ key: "verdict", dir: "asc" }}
      rowSeverity={severity}
    />
  );
}
