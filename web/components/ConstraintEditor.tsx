"use client";

import { useState } from "react";
import { rerunAgent } from "@/lib/api";

export interface ConstraintsBlob {
  price_ladder: number[];
  promo_states: number[];
  cog_pct: number;
  margin_floor_pct: number;
  comp_gap_pct: number;
  max_decrease: number;
  max_increase: number;
  objective: "revenue" | "margin";
}

interface Props {
  runId: string;
  current: ConstraintsBlob;
}

function parseLadder(text: string): number[] | null {
  const parts = text
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (!parts.length) return null;
  const nums = parts.map((p) => Number(p));
  if (nums.some((n) => !Number.isFinite(n) || n <= 0)) return null;
  return nums.sort((a, b) => a - b);
}

export function ConstraintEditor({ runId, current }: Props) {
  const [objective, setObjective] = useState<"revenue" | "margin">(current.objective);
  const [ladder, setLadder] = useState<string>(current.price_ladder.join(", "));
  const [marginFloor, setMarginFloor] = useState<string>(String(current.margin_floor_pct));
  const [compGap, setCompGap] = useState<string>(String(current.comp_gap_pct));
  const [maxDec, setMaxDec] = useState<string>(String(current.max_decrease));
  const [maxInc, setMaxInc] = useState<string>(String(current.max_increase));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setError(null);
    const ladderNums = parseLadder(ladder);
    if (!ladderNums) {
      setError("Price ladder must be a comma-separated list of positive numbers.");
      return;
    }
    const payload: Record<string, unknown> = {
      objective,
      price_ladder: ladderNums,
      margin_floor_pct: Number(marginFloor),
      comp_gap_pct: Number(compGap),
      max_decrease: Number(maxDec),
      max_increase: Number(maxInc),
    };
    for (const k of Object.keys(payload)) {
      const v = payload[k];
      if (typeof v === "number" && !Number.isFinite(v)) {
        setError(`${k} must be a finite number.`);
        return;
      }
    }
    setBusy(true);
    try {
      await rerunAgent(runId, "optimization", payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3 rounded-md border border-slate-800 bg-slate-900/40 p-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label className="space-y-1">
          <span className="block text-[10px] uppercase tracking-wider text-slate-500">Objective</span>
          <select
            value={objective}
            onChange={(e) => setObjective(e.target.value as "revenue" | "margin")}
            className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200"
          >
            <option value="revenue">revenue</option>
            <option value="margin">margin</option>
          </select>
        </label>
        <label className="space-y-1 sm:col-span-2">
          <span className="block text-[10px] uppercase tracking-wider text-slate-500">
            Price ladder · multipliers of base price
          </span>
          <input
            value={ladder}
            onChange={(e) => setLadder(e.target.value)}
            className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-xs text-slate-200"
            placeholder="0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15"
          />
        </label>
        <label className="space-y-1">
          <span className="block text-[10px] uppercase tracking-wider text-slate-500">Margin floor (fraction of base)</span>
          <input
            type="number"
            step="0.01"
            value={marginFloor}
            onChange={(e) => setMarginFloor(e.target.value)}
            className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-xs text-slate-200"
          />
        </label>
        <label className="space-y-1">
          <span className="block text-[10px] uppercase tracking-wider text-slate-500">Competitor gap (± fraction)</span>
          <input
            type="number"
            step="0.01"
            value={compGap}
            onChange={(e) => setCompGap(e.target.value)}
            className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-xs text-slate-200"
          />
        </label>
        <label className="space-y-1">
          <span className="block text-[10px] uppercase tracking-wider text-slate-500">Max decrease</span>
          <input
            type="number"
            step="0.01"
            value={maxDec}
            onChange={(e) => setMaxDec(e.target.value)}
            className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-xs text-slate-200"
          />
        </label>
        <label className="space-y-1">
          <span className="block text-[10px] uppercase tracking-wider text-slate-500">Max increase</span>
          <input
            type="number"
            step="0.01"
            value={maxInc}
            onChange={(e) => setMaxInc(e.target.value)}
            className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-xs text-slate-200"
          />
        </label>
      </div>
      {error && <p className="text-[11px] text-rose-300">{error}</p>}
      <div className="flex items-center justify-between">
        <p className="text-[10px] text-slate-500">
          Submitting re-solves the MILP and re-arms the approval gate.
        </p>
        <button
          type="button"
          onClick={submit}
          disabled={busy}
          className="rounded bg-emerald-500/20 px-3 py-1 text-xs font-medium text-emerald-200 hover:bg-emerald-500/30 disabled:cursor-wait disabled:opacity-50"
        >
          {busy ? "Re-solving…" : "Save & re-solve"}
        </button>
      </div>
    </div>
  );
}
