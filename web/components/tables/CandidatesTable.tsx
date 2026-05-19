"use client";

import { useState } from "react";

export interface CandidateAttempt {
  model: string;
  own_elasticity: number;
  std_err: number;
  r_squared: number;
  n_obs: number;
  sign_ok: boolean;
  diagnostics: {
    test_wape?: number;
    train_wape?: number;
    n_test?: number;
  };
}

export interface CandidatesRow {
  ppg_id: string;
  winner_model: string;
  sign_retry_fired: boolean;
  n_train: number;
  n_test: number;
  attempts: CandidateAttempt[];
}

const MODEL_LABEL: Record<string, string> = {
  loglog_ols: "log-log OLS",
  semilog_ols: "semi-log OLS",
  lightgbm: "LightGBM",
};

function fmt(n: number | undefined | null, digits = 3): string {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  if (!Number.isFinite(n)) return "∞";
  return n.toFixed(digits);
}

export interface CandidatesTableProps {
  rows: CandidatesRow[];
  selectedPpg: string | null;
  onSelectPpg: (ppg: string) => void;
}

export function CandidatesTable({ rows, selectedPpg, onSelectPpg }: CandidatesTableProps) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  function toggle(ppg: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(ppg) ? next.delete(ppg) : next.add(ppg);
      return next;
    });
  }
  return (
    <div className="overflow-hidden rounded border border-slate-800">
      <table className="w-full text-[11px]">
        <thead className="bg-slate-900/80 text-[10px] uppercase tracking-wider text-slate-500">
          <tr>
            <th className="w-8" />
            <th className="px-2 py-1.5 text-left">PPG</th>
            <th className="px-2 py-1.5 text-left">Winner</th>
            <th className="px-2 py-1.5 text-right">Elasticity</th>
            <th className="px-2 py-1.5 text-right">R²</th>
            <th className="px-2 py-1.5 text-right">Test WAPE</th>
            <th className="px-2 py-1.5 text-right">N train</th>
            <th className="px-2 py-1.5 text-center">Sign OK</th>
            <th className="px-2 py-1.5 text-center">Retry</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {rows.map((row) => {
            const winnerAttempt = row.attempts.find((a) => a.model === row.winner_model);
            const isExpanded = expanded.has(row.ppg_id);
            const isSelected = selectedPpg === row.ppg_id;
            return (
              <FragmentRow
                key={row.ppg_id}
                row={row}
                winnerAttempt={winnerAttempt}
                isExpanded={isExpanded}
                isSelected={isSelected}
                onToggle={() => toggle(row.ppg_id)}
                onSelect={() => onSelectPpg(row.ppg_id)}
              />
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

interface FragmentRowProps {
  row: CandidatesRow;
  winnerAttempt: CandidateAttempt | undefined;
  isExpanded: boolean;
  isSelected: boolean;
  onToggle: () => void;
  onSelect: () => void;
}

function FragmentRow({ row, winnerAttempt, isExpanded, isSelected, onToggle, onSelect }: FragmentRowProps) {
  return (
    <>
      <tr
        onClick={onSelect}
        className={`cursor-pointer ${isSelected ? "bg-emerald-500/5" : "hover:bg-slate-900/60"}`}
      >
        <td className="px-1 text-center">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onToggle();
            }}
            className="text-slate-500 hover:text-slate-300"
            aria-label={isExpanded ? "collapse" : "expand"}
          >
            {isExpanded ? "▾" : "▸"}
          </button>
        </td>
        <td className="px-2 py-1.5 font-mono text-slate-200">{row.ppg_id}</td>
        <td className="px-2 py-1.5 text-slate-200">{MODEL_LABEL[row.winner_model] ?? row.winner_model}</td>
        <td className="px-2 py-1.5 text-right font-mono text-slate-200">
          {winnerAttempt ? fmt(winnerAttempt.own_elasticity, 2) : "—"}
        </td>
        <td className="px-2 py-1.5 text-right font-mono text-slate-300">
          {winnerAttempt ? fmt(winnerAttempt.r_squared, 2) : "—"}
        </td>
        <td className="px-2 py-1.5 text-right font-mono text-slate-300">
          {winnerAttempt ? fmt(winnerAttempt.diagnostics.test_wape, 3) : "—"}
        </td>
        <td className="px-2 py-1.5 text-right font-mono text-slate-400">{row.n_train}</td>
        <td className="px-2 py-1.5 text-center">
          {winnerAttempt?.sign_ok ? (
            <span className="rounded border border-emerald-500/40 bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">
              ✓
            </span>
          ) : (
            <span className="rounded border border-rose-500/40 bg-rose-500/15 px-1.5 py-0.5 text-[10px] text-rose-300">
              ✗
            </span>
          )}
        </td>
        <td className="px-2 py-1.5 text-center text-slate-400">
          {row.sign_retry_fired ? "yes" : "—"}
        </td>
      </tr>
      {isExpanded && (
        <tr className="bg-slate-950/40">
          <td colSpan={9} className="px-3 py-2">
            <AttemptsList row={row} />
          </td>
        </tr>
      )}
    </>
  );
}

function AttemptsList({ row }: { row: CandidatesRow }) {
  return (
    <div className="space-y-1">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">
        All candidates · sorted as fitted
      </div>
      <table className="w-full text-[10.5px]">
        <thead className="text-[9.5px] uppercase tracking-wider text-slate-600">
          <tr>
            <th className="px-2 py-1 text-left">Model</th>
            <th className="px-2 py-1 text-right">Elasticity</th>
            <th className="px-2 py-1 text-right">Std err</th>
            <th className="px-2 py-1 text-right">R²</th>
            <th className="px-2 py-1 text-right">Train WAPE</th>
            <th className="px-2 py-1 text-right">Test WAPE</th>
            <th className="px-2 py-1 text-center">Sign</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60">
          {row.attempts.map((a) => {
            const isWinner = a.model === row.winner_model;
            return (
              <tr key={a.model} className={isWinner ? "bg-emerald-500/5 text-slate-200" : "text-slate-400"}>
                <td className="px-2 py-1 font-mono">
                  {MODEL_LABEL[a.model] ?? a.model}
                  {isWinner && (
                    <span className="ml-2 rounded border border-emerald-500/40 px-1 text-[9px] text-emerald-300">
                      winner
                    </span>
                  )}
                </td>
                <td className="px-2 py-1 text-right font-mono">{fmt(a.own_elasticity, 2)}</td>
                <td className="px-2 py-1 text-right font-mono">{fmt(a.std_err, 3)}</td>
                <td className="px-2 py-1 text-right font-mono">{fmt(a.r_squared, 2)}</td>
                <td className="px-2 py-1 text-right font-mono">{fmt(a.diagnostics.train_wape, 3)}</td>
                <td className="px-2 py-1 text-right font-mono">{fmt(a.diagnostics.test_wape, 3)}</td>
                <td className="px-2 py-1 text-center">
                  {a.sign_ok ? (
                    <span className="text-emerald-300">✓</span>
                  ) : (
                    <span className="text-rose-300">✗</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
