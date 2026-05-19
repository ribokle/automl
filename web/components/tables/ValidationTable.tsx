"use client";

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

function fmtPct(v: number): string {
  if (!Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(0)}%`;
}

function fmt(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

export function ValidationTable({ rows }: { rows: ValidationRow[] }) {
  if (rows.length === 0) {
    return <p className="text-[11px] text-slate-500">No validation results.</p>;
  }
  return (
    <div className="overflow-x-auto rounded border border-slate-800">
      <table className="min-w-full text-[11px]">
        <thead className="bg-slate-900/80 text-[10px] uppercase tracking-wider text-slate-500">
          <tr>
            <th className="px-2 py-1.5 text-left">PPG</th>
            <th className="px-2 py-1.5 text-center">Verdict</th>
            <th className="px-2 py-1.5 text-right">Sign stab.</th>
            <th className="px-2 py-1.5 text-right">WAPE</th>
            <th className="px-2 py-1.5 text-right">ε mean</th>
            <th className="px-2 py-1.5 text-right">ε CV</th>
            <th className="px-2 py-1.5 text-right">Folds</th>
            <th className="px-2 py-1.5 text-left">Winner</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/70 font-mono">
          {rows.map((r) => (
            <tr key={r.ppg_id} className="hover:bg-slate-900/40">
              <td className="px-2 py-1.5 text-slate-200">{r.ppg_id}</td>
              <td className="px-2 py-1.5 text-center">
                <span className={`rounded border px-1.5 py-0.5 text-[9.5px] uppercase ${VERDICT_STYLE[r.verdict]}`}>
                  {r.verdict}
                </span>
              </td>
              <td className="px-2 py-1.5 text-right text-slate-300">{fmtPct(r.sign_stability)}</td>
              <td className="px-2 py-1.5 text-right text-slate-300">{fmt(r.wape_mean, 3)}</td>
              <td className="px-2 py-1.5 text-right text-slate-300">{fmt(r.elasticity_mean)}</td>
              <td className="px-2 py-1.5 text-right text-slate-300">{fmt(r.elasticity_cv)}</td>
              <td className="px-2 py-1.5 text-right text-slate-400">{r.n_folds}</td>
              <td className="px-2 py-1.5 text-slate-400">{r.winner}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
