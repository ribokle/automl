export interface QualityData {
  summary: { pass: number; warn: number; fail: number };
  checks: {
    source: "dbt" | "ge";
    name: string;
    status: "pass" | "warn" | "fail";
    severity: string;
    message: string;
    failing_rows: number | null;
  }[];
}

const TONE: Record<QualityData["checks"][number]["status"], string> = {
  pass: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  warn: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  fail: "bg-rose-500/15 text-rose-300 border-rose-500/30",
};

export function QualityPanel({ data }: { data: QualityData }) {
  const { pass, warn, fail } = data.summary;
  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-2 text-[11px]">
        <span className="rounded border border-emerald-500/40 bg-emerald-500/15 px-2 py-0.5 text-emerald-300">
          {pass} pass
        </span>
        {warn > 0 && (
          <span className="rounded border border-amber-500/40 bg-amber-500/15 px-2 py-0.5 text-amber-300">
            {warn} warn
          </span>
        )}
        {fail > 0 && (
          <span className="rounded border border-rose-500/40 bg-rose-500/15 px-2 py-0.5 text-rose-300">
            {fail} fail
          </span>
        )}
      </div>
      <ul className="max-h-72 space-y-1 overflow-y-auto pr-1 text-[11px]">
        {data.checks.map((c, i) => (
          <li
            key={`${c.source}:${c.name}:${i}`}
            className="flex items-start justify-between gap-2 rounded border border-slate-800 bg-slate-900/40 px-2 py-1"
          >
            <div className="min-w-0">
              <div className="font-mono text-[10px] uppercase tracking-wide text-slate-500">{c.source}</div>
              <div className="truncate font-mono text-slate-300">{c.name}</div>
              {(c.message || c.failing_rows) && (
                <div className="text-[10px] text-slate-500">
                  {c.message}
                  {c.failing_rows ? ` · ${c.failing_rows} rows` : ""}
                </div>
              )}
            </div>
            <span className={`shrink-0 rounded border px-1.5 py-0.5 text-[10px] ${TONE[c.status]}`}>{c.status}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
