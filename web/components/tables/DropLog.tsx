export interface DropLogData {
  dropped: { feature: string; reason: string }[];
  kept: string[];
}

export function DropLog({ data }: { data: DropLogData }) {
  if (!data.dropped.length) {
    return <p className="text-[11px] text-slate-500">No features dropped — all candidates passed VIF + corr thresholds.</p>;
  }
  return (
    <ul className="space-y-1 text-[11px]">
      {data.dropped.map((d, i) => (
        <li
          key={i}
          className="flex items-center justify-between gap-2 rounded border border-slate-800 bg-slate-900/40 px-2 py-1"
        >
          <span className="font-mono text-rose-300">{d.feature}</span>
          <span className="text-slate-400">{d.reason}</span>
        </li>
      ))}
    </ul>
  );
}

export function KeptList({ data }: { data: DropLogData }) {
  if (!data.kept.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5 text-[11px]">
      {data.kept.map((k) => (
        <span
          key={k}
          className="rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 font-mono text-emerald-300"
        >
          {k}
        </span>
      ))}
    </div>
  );
}
