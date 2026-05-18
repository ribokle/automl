export interface FindingsBlob {
  summary: string;
  anomalies: { tag: string; severity: "info" | "warn" | "error"; message: string }[];
  recommendations: string[];
}

const TONE: Record<FindingsBlob["anomalies"][number]["severity"], string> = {
  info: "bg-slate-700/40 text-slate-300 border-slate-600",
  warn: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  error: "bg-rose-500/15 text-rose-300 border-rose-500/30",
};

export function AnomalyTable({ data }: { data: FindingsBlob }) {
  if (!data.anomalies.length) {
    return <p className="text-[11px] text-slate-500">No anomalies flagged.</p>;
  }
  return (
    <ul className="space-y-1 text-[11px]">
      {data.anomalies.map((a, i) => (
        <li
          key={i}
          className="flex items-start justify-between gap-2 rounded border border-slate-800 bg-slate-900/40 px-2 py-1"
        >
          <div className="min-w-0">
            <div className="font-mono text-[10px] uppercase tracking-wide text-slate-500">{a.tag}</div>
            <div className="text-slate-300">{a.message}</div>
          </div>
          <span className={`shrink-0 rounded border px-1.5 py-0.5 text-[10px] ${TONE[a.severity]}`}>{a.severity}</span>
        </li>
      ))}
    </ul>
  );
}
