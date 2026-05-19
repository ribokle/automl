"use client";

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

function fmtCurrency(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (Math.abs(v) >= 1_000) return `$${(v / 1_000).toFixed(1)}k`;
  return `$${v.toFixed(2)}`;
}

function fmtPct(mult: number): string {
  const delta = mult - 1;
  const sign = delta > 0 ? "+" : "";
  return `${sign}${(delta * 100).toFixed(1)}%`;
}

export function RecommendationTable({ rows }: { rows: RecommendationRow[] }) {
  if (rows.length === 0) {
    return <p className="text-[11px] text-slate-500">No optimisation results.</p>;
  }
  return (
    <div className="overflow-x-auto rounded border border-slate-800">
      <table className="min-w-full text-[11px]">
        <thead className="bg-slate-900/80 text-[10px] uppercase tracking-wider text-slate-500">
          <tr>
            <th className="px-2 py-1.5 text-left">PPG</th>
            <th className="px-2 py-1.5 text-right">Base</th>
            <th className="px-2 py-1.5 text-right">Recommended</th>
            <th className="px-2 py-1.5 text-right">Δ</th>
            <th className="px-2 py-1.5 text-center">Promo</th>
            <th className="px-2 py-1.5 text-right">Units</th>
            <th className="px-2 py-1.5 text-right">Revenue</th>
            <th className="px-2 py-1.5 text-right">Margin</th>
            <th className="px-2 py-1.5 text-center">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/70 font-mono">
          {rows.map((r) => (
            <tr key={r.ppg_id} className="hover:bg-slate-900/40">
              <td className="px-2 py-1.5 text-slate-200">{r.ppg_id}</td>
              <td className="px-2 py-1.5 text-right text-slate-400">{fmtCurrency(r.base_price)}</td>
              <td className="px-2 py-1.5 text-right text-slate-100">{fmtCurrency(r.price)}</td>
              <td className={`px-2 py-1.5 text-right ${r.price_multiplier > 1 ? "text-emerald-300" : r.price_multiplier < 1 ? "text-amber-300" : "text-slate-300"}`}>
                {fmtPct(r.price_multiplier)}
              </td>
              <td className="px-2 py-1.5 text-center text-slate-300">{r.promo ? "on" : "off"}</td>
              <td className="px-2 py-1.5 text-right text-slate-300">{r.units.toFixed(0)}</td>
              <td className="px-2 py-1.5 text-right text-slate-300">{fmtCurrency(r.revenue)}</td>
              <td className="px-2 py-1.5 text-right text-slate-300">{fmtCurrency(r.margin)}</td>
              <td className="px-2 py-1.5 text-center">
                {r.relaxed ? (
                  <span className="rounded border border-amber-500/40 bg-amber-500/15 px-1.5 py-0.5 text-[9.5px] text-amber-300">
                    relaxed
                  </span>
                ) : r.feasible_strict ? (
                  <span className="rounded border border-emerald-500/40 bg-emerald-500/15 px-1.5 py-0.5 text-[9.5px] text-emerald-300">
                    feasible
                  </span>
                ) : (
                  <span className="text-slate-500">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
