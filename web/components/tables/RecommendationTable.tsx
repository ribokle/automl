"use client";

import { ResultsTable, type ColumnDef, type Severity } from "./ResultsTable";

export interface RecommendationRow {
  ppg_id: string;
  objective: string;
  price_multiplier: number;
  price: number;
  base_price: number;
  promo: number;
  units: number;
  revenue: number;
  margin: number;
  feasible_strict: boolean;
  relaxed: boolean;
  model_kind: string;
  rationale?: string;
}

function fmtCurrency(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (Math.abs(v) >= 1_000) return `$${(v / 1_000).toFixed(1)}k`;
  return `$${v.toFixed(2)}`;
}

function deltaPct(mult: number): { text: string; positive: boolean; zero: boolean } {
  const d = mult - 1;
  const sign = d > 0 ? "+" : "";
  return { text: `${sign}${(d * 100).toFixed(1)}%`, positive: d > 0, zero: d === 0 };
}

const COLUMNS: ColumnDef<RecommendationRow>[] = [
  { key: "ppg_id", label: "PPG" },
  {
    key: "base_price",
    label: "Base",
    numeric: true,
    format: (r) => <span className="text-slate-400">{fmtCurrency(r.base_price)}</span>,
  },
  {
    key: "price",
    label: "Recommended",
    numeric: true,
    format: (r) => <span className="text-slate-100">{fmtCurrency(r.price)}</span>,
  },
  {
    key: "price_multiplier",
    label: "Δ",
    numeric: true,
    format: (r) => {
      const d = deltaPct(r.price_multiplier);
      const color = d.zero
        ? "text-slate-300"
        : d.positive
          ? "text-emerald-300"
          : "text-amber-300";
      return <span className={color}>{d.text}</span>;
    },
  },
  {
    key: "promo",
    label: "Promo",
    align: "center",
    format: (r) => (r.promo ? "on" : "off"),
  },
  {
    key: "units",
    label: "Units",
    numeric: true,
    format: (r) => r.units.toFixed(0),
  },
  { key: "revenue", label: "Revenue", numeric: true, format: (r) => fmtCurrency(r.revenue) },
  { key: "margin", label: "Margin", numeric: true, format: (r) => fmtCurrency(r.margin) },
  {
    key: "status",
    label: "Status",
    align: "center",
    sortValue: (r) => (r.relaxed ? 1 : r.feasible_strict ? 0 : 2),
    format: (r) =>
      r.relaxed ? (
        <span className="rounded border border-amber-500/40 bg-amber-500/15 px-1.5 py-0.5 text-[9.5px] text-amber-300">
          relaxed
        </span>
      ) : r.feasible_strict ? (
        <span className="rounded border border-emerald-500/40 bg-emerald-500/15 px-1.5 py-0.5 text-[9.5px] text-emerald-300">
          feasible
        </span>
      ) : (
        <span className="text-slate-500">—</span>
      ),
  },
];

function severity(row: RecommendationRow): Severity | null {
  if (row.relaxed) return "warn";
  if (row.feasible_strict) return "pass";
  return null;
}

export function RecommendationTable({ rows }: { rows: RecommendationRow[] }) {
  return (
    <ResultsTable
      rows={rows}
      columns={COLUMNS}
      rowKey={(r) => r.ppg_id}
      empty="No optimisation results."
      defaultSort={{ key: "revenue", dir: "desc" }}
      rowSeverity={severity}
    />
  );
}
