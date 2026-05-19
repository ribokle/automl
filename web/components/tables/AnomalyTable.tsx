"use client";

import { ResultsTable, type ColumnDef, type Severity } from "./ResultsTable";

export interface FindingsBlob {
  summary: string;
  anomalies: { tag: string; severity: "info" | "warn" | "error"; message: string }[];
  recommendations: string[];
}

type Anomaly = FindingsBlob["anomalies"][number];

const SEVERITY_PILL: Record<Anomaly["severity"], string> = {
  info: "border-slate-600 bg-slate-700/40 text-slate-300",
  warn: "border-amber-500/30 bg-amber-500/15 text-amber-300",
  error: "border-rose-500/30 bg-rose-500/15 text-rose-300",
};

const SEVERITY_RANK: Record<Anomaly["severity"], number> = { error: 0, warn: 1, info: 2 };

const SEVERITY_ROW: Record<Anomaly["severity"], Severity> = {
  info: "info",
  warn: "warn",
  error: "fail",
};

const COLUMNS: ColumnDef<Anomaly>[] = [
  {
    key: "tag",
    label: "Tag",
    format: (r) => (
      <span className="font-mono text-[10px] uppercase tracking-wide text-slate-400">
        {r.tag}
      </span>
    ),
  },
  {
    key: "severity",
    label: "Severity",
    align: "center",
    sortValue: (r) => SEVERITY_RANK[r.severity],
    format: (r) => (
      <span
        className={`rounded border px-1.5 py-0.5 text-[9.5px] uppercase ${SEVERITY_PILL[r.severity]}`}
      >
        {r.severity}
      </span>
    ),
  },
  {
    key: "message",
    label: "Message",
    format: (r) => <span className="text-slate-300">{r.message}</span>,
    unsortable: true,
  },
];

export function AnomalyTable({ data }: { data: FindingsBlob }) {
  return (
    <ResultsTable
      rows={data.anomalies}
      columns={COLUMNS}
      rowKey={(r) => `${r.tag}:${r.message}`}
      empty="No anomalies flagged."
      defaultSort={{ key: "severity", dir: "asc" }}
      rowSeverity={(r) => SEVERITY_ROW[r.severity]}
      stickyFirst={false}
    />
  );
}
