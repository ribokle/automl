import type { ProfileBlob } from "./DataPreview";

const IDENT_COLUMNS = new Set(["sku", "store_id", "ppg_id", "brand", "category", "pack_size"]);
const TEMPORAL_COLUMNS = new Set(["week_start", "holiday"]);
const TARGET_COLUMNS = new Set(["units"]);
const FLAG_SUFFIXES = ["_flag", "_pct"];

function roleFor(name: string, dtype: string | undefined): { role: string; tone: string } {
  if (TARGET_COLUMNS.has(name)) return { role: "target", tone: "bg-emerald-500/15 text-emerald-300" };
  if (IDENT_COLUMNS.has(name)) return { role: "identifier", tone: "bg-slate-700/40 text-slate-300" };
  if (TEMPORAL_COLUMNS.has(name)) return { role: "temporal", tone: "bg-violet-500/15 text-violet-300" };
  if (FLAG_SUFFIXES.some((s) => name.endsWith(s))) return { role: "flag", tone: "bg-amber-500/15 text-amber-300" };
  const t = (dtype ?? "").toUpperCase();
  if (t.includes("INT") || t.includes("DOUBLE") || t.includes("DECIMAL") || t.includes("FLOAT")) {
    return { role: "numeric", tone: "bg-sky-500/15 text-sky-300" };
  }
  return { role: "other", tone: "bg-slate-700/40 text-slate-300" };
}

export function SchemaTable({ data }: { data: ProfileBlob }) {
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
          {data.profile.columns.map((c) => {
            const r = roleFor(c.name, c.dtype);
            return (
              <tr key={c.name} className="border-t border-slate-800">
                <td className="px-2 py-1 font-mono text-slate-300">{c.name}</td>
                <td className="px-2 py-1 font-mono text-slate-500">{c.dtype ?? ""}</td>
                <td className="px-2 py-1">
                  <span className={`rounded px-1.5 py-0.5 text-[10px] ${r.tone}`}>{r.role}</span>
                </td>
                <td className="px-2 py-1 text-right font-mono text-slate-400">
                  {typeof c.null_pct === "number" ? `${c.null_pct.toFixed(1)}%` : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
