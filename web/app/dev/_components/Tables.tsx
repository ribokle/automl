import { ANOMALIES, PREVIEW_ROWS, QUALITY_CHECKS, SCHEMA, DROPPED } from "../_data";

export function DataPreview() {
  return (
    <div className="overflow-hidden rounded border border-slate-800">
      <table className="w-full text-[11px]">
        <thead className="bg-slate-900/80 text-slate-400">
          <tr>
            <th className="px-2 py-1 text-left">sku</th>
            <th className="px-2 py-1 text-left">week</th>
            <th className="px-2 py-1 text-left">store</th>
            <th className="px-2 py-1 text-right">units</th>
            <th className="px-2 py-1 text-right">price</th>
            <th className="px-2 py-1 text-right">base</th>
            <th className="px-2 py-1 text-right">tpr</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {PREVIEW_ROWS.map((r, i) => (
            <tr key={i} className="border-t border-slate-800 text-slate-300">
              <td className="px-2 py-1">{r.sku}</td>
              <td className="px-2 py-1">{r.week}</td>
              <td className="px-2 py-1">{r.store}</td>
              <td className="px-2 py-1 text-right">{r.units}</td>
              <td className="px-2 py-1 text-right">{r.price.toFixed(2)}</td>
              <td className="px-2 py-1 text-right">{r.base_price.toFixed(2)}</td>
              <td className="px-2 py-1 text-right">{r.tpr}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function SchemaTable() {
  const roleTone: Record<string, string> = {
    target: "bg-emerald-500/15 text-emerald-300",
    numeric: "bg-sky-500/15 text-sky-300",
    flag: "bg-amber-500/15 text-amber-300",
    identifier: "bg-slate-700/40 text-slate-300",
    temporal: "bg-violet-500/15 text-violet-300",
  };
  return (
    <div className="overflow-hidden rounded border border-slate-800">
      <table className="w-full text-[11px]">
        <thead className="bg-slate-900/80 text-slate-400">
          <tr>
            <th className="px-2 py-1 text-left">column</th>
            <th className="px-2 py-1 text-left">dtype</th>
            <th className="px-2 py-1 text-left">role</th>
            <th className="px-2 py-1 text-right">null %</th>
          </tr>
        </thead>
        <tbody>
          {SCHEMA.map((s) => (
            <tr key={s.column} className="border-t border-slate-800">
              <td className="px-2 py-1 font-mono text-slate-300">{s.column}</td>
              <td className="px-2 py-1 font-mono text-slate-500">{s.dtype}</td>
              <td className="px-2 py-1">
                <span className={`rounded px-1.5 py-0.5 text-[10px] ${roleTone[s.role]}`}>{s.role}</span>
              </td>
              <td className="px-2 py-1 text-right font-mono text-slate-400">{s.nulls}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function QualityPanel() {
  const tone: Record<string, string> = {
    pass: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    warn: "bg-amber-500/15 text-amber-300 border-amber-500/30",
    fail: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  };
  const passN = QUALITY_CHECKS.filter((c) => c.status === "pass").length;
  const warnN = QUALITY_CHECKS.filter((c) => c.status === "warn").length;
  const failN = QUALITY_CHECKS.filter((c) => c.status === "fail").length;
  return (
    <div>
      <div className="mb-2 flex gap-2 text-[11px]">
        <span className="rounded border border-emerald-500/40 bg-emerald-500/15 px-2 py-0.5 text-emerald-300">
          {passN} pass
        </span>
        {warnN > 0 && (
          <span className="rounded border border-amber-500/40 bg-amber-500/15 px-2 py-0.5 text-amber-300">
            {warnN} warn
          </span>
        )}
        {failN > 0 && (
          <span className="rounded border border-rose-500/40 bg-rose-500/15 px-2 py-0.5 text-rose-300">
            {failN} fail
          </span>
        )}
      </div>
      <ul className="space-y-1 text-[11px]">
        {QUALITY_CHECKS.map((c, i) => (
          <li
            key={i}
            className="flex items-start justify-between gap-2 rounded border border-slate-800 bg-slate-900/40 px-2 py-1"
          >
            <div className="min-w-0">
              <div className="font-mono text-[10px] text-slate-500">{c.source}</div>
              <div className="truncate font-mono text-slate-300">{c.name}</div>
              {c.message && <div className="text-[10px] text-slate-500">{c.message}</div>}
            </div>
            <span className={`shrink-0 rounded border px-1.5 py-0.5 text-[10px] ${tone[c.status]}`}>
              {c.status}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function AnomalyTable() {
  return (
    <div className="overflow-hidden rounded border border-slate-800">
      <table className="w-full text-[11px]">
        <thead className="bg-slate-900/80 text-slate-400">
          <tr>
            <th className="px-2 py-1 text-left">sku</th>
            <th className="px-2 py-1 text-left">week</th>
            <th className="px-2 py-1 text-right">units</th>
            <th className="px-2 py-1 text-right">price</th>
            <th className="px-2 py-1 text-left">reason</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {ANOMALIES.map((a, i) => (
            <tr key={i} className="border-t border-slate-800 text-slate-300">
              <td className="px-2 py-1">{a.sku}</td>
              <td className="px-2 py-1">{a.week}</td>
              <td className="px-2 py-1 text-right">{a.units}</td>
              <td className="px-2 py-1 text-right">{a.price.toFixed(2)}</td>
              <td className="px-2 py-1 text-amber-300">{a.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DropLog() {
  return (
    <ul className="space-y-1 text-[11px]">
      {DROPPED.map((d, i) => (
        <li key={i} className="flex items-center justify-between rounded border border-slate-800 bg-slate-900/40 px-2 py-1">
          <span className="font-mono text-rose-300">{d.feature}</span>
          <span className="text-slate-400">{d.reason}</span>
        </li>
      ))}
    </ul>
  );
}
