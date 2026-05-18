export interface ProfileBlob {
  profile: {
    row_count: number;
    columns: { name: string; type: string; null_pct: number }[];
  };
  sample: Record<string, unknown>[];
  outliers: { column: string; n_outliers: number; outlier_pct: number }[];
}

export function DataPreview({ data }: { data: ProfileBlob }) {
  const rows = data.sample.slice(0, 10);
  if (!rows.length) {
    return <p className="text-xs text-slate-500">No sample rows.</p>;
  }
  const cols = Object.keys(rows[0]);
  return (
    <div className="overflow-x-auto rounded border border-slate-800">
      <table className="w-full text-[11px]">
        <thead className="bg-slate-900/80 text-slate-400">
          <tr>
            {cols.map((c) => (
              <th key={c} className="px-2 py-1 text-left font-mono">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="font-mono">
          {rows.map((row, i) => (
            <tr key={i} className="border-t border-slate-800 text-slate-300">
              {cols.map((c) => (
                <td key={c} className="px-2 py-1">
                  {String(row[c] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
