export interface TargetRelationshipRow {
  feature: string;
  spearman: number;
  pearson?: number;
  n?: number;
}

export function TargetRelationship({ rows }: { rows: TargetRelationshipRow[] }) {
  if (!rows.length) return <p className="text-[11px] text-slate-500">No candidates ranked.</p>;
  return (
    <div className="overflow-hidden rounded border border-slate-800">
      <table className="w-full text-[11px]">
        <thead className="bg-slate-900/80 text-slate-400">
          <tr>
            <th className="px-2 py-1 text-left">feature</th>
            <th className="px-2 py-1 text-right">spearman ρ</th>
            {rows[0].pearson !== undefined && (
              <th className="px-2 py-1 text-right">pearson r</th>
            )}
            {rows[0].n !== undefined && <th className="px-2 py-1 text-right">n</th>}
          </tr>
        </thead>
        <tbody className="font-mono">
          {rows.map((r) => {
            const mag = Math.abs(r.spearman);
            const tone =
              mag >= 0.6
                ? "text-emerald-300"
                : mag >= 0.3
                ? "text-sky-300"
                : "text-slate-400";
            return (
              <tr key={r.feature} className="border-t border-slate-800 text-slate-300">
                <td className="px-2 py-1">{r.feature}</td>
                <td className={`px-2 py-1 text-right tabular-nums ${tone}`}>
                  {r.spearman >= 0 ? "+" : ""}
                  {r.spearman.toFixed(3)}
                </td>
                {r.pearson !== undefined && (
                  <td className="px-2 py-1 text-right tabular-nums text-slate-400">
                    {r.pearson >= 0 ? "+" : ""}
                    {r.pearson.toFixed(3)}
                  </td>
                )}
                {r.n !== undefined && (
                  <td className="px-2 py-1 text-right tabular-nums text-slate-500">{r.n}</td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
