"use client";

export interface HolidayLiftRow {
  ppg_id: string;
  week_start: string;
  holiday: string;
  units: number;
  trailing_median: number;
  lift: number;
}

export function HolidayLiftTable({ rows }: { rows: HolidayLiftRow[] }) {
  if (!rows.length) {
    return <p className="text-[11px] text-slate-500">No holiday weeks in this panel.</p>;
  }
  const truncated = rows.slice(0, 50);
  return (
    <div className="max-h-96 overflow-y-auto rounded border border-slate-800">
      <table className="w-full text-[11px]">
        <thead className="sticky top-0 bg-slate-900 text-slate-400">
          <tr>
            <th className="px-2 py-1 text-left">PPG</th>
            <th className="px-2 py-1 text-left">Week</th>
            <th className="px-2 py-1 text-left">Holiday</th>
            <th className="px-2 py-1 text-right">Units</th>
            <th className="px-2 py-1 text-right">Trailing med</th>
            <th className="px-2 py-1 text-right">Lift</th>
          </tr>
        </thead>
        <tbody>
          {truncated.map((r, i) => {
            const liftColour =
              r.lift > 0.2 ? "text-emerald-400" : r.lift < -0.2 ? "text-rose-400" : "text-slate-300";
            return (
              <tr key={i} className="border-t border-slate-800 hover:bg-slate-900/40">
                <td className="px-2 py-1 font-mono">{r.ppg_id}</td>
                <td className="px-2 py-1 text-slate-400">{r.week_start.split("T")[0]}</td>
                <td className="px-2 py-1">{r.holiday}</td>
                <td className="px-2 py-1 text-right">{r.units.toLocaleString()}</td>
                <td className="px-2 py-1 text-right">{r.trailing_median.toFixed(0)}</td>
                <td className={`px-2 py-1 text-right ${liftColour}`}>
                  {r.lift >= 0 ? "+" : ""}
                  {(r.lift * 100).toFixed(1)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows.length > truncated.length && (
        <p className="border-t border-slate-800 px-2 py-1 text-[10px] text-slate-500">
          showing top {truncated.length} of {rows.length} by absolute lift
        </p>
      )}
    </div>
  );
}
