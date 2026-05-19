"use client";

import { ResultsTable, type ColumnDef } from "./ResultsTable";

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

function winnerAttempt(row: CandidatesRow): CandidateAttempt | undefined {
  return row.attempts.find((a) => a.model === row.winner_model);
}

const COLUMNS: ColumnDef<CandidatesRow>[] = [
  { key: "ppg_id", label: "PPG" },
  {
    key: "winner_model",
    label: "Winner",
    format: (r) => MODEL_LABEL[r.winner_model] ?? r.winner_model,
  },
  {
    key: "elasticity",
    label: "Elasticity",
    numeric: true,
    sortValue: (r) => winnerAttempt(r)?.own_elasticity ?? null,
    format: (r) => fmt(winnerAttempt(r)?.own_elasticity, 2),
  },
  {
    key: "r_squared",
    label: "R²",
    numeric: true,
    sortValue: (r) => winnerAttempt(r)?.r_squared ?? null,
    format: (r) => fmt(winnerAttempt(r)?.r_squared, 2),
  },
  {
    key: "test_wape",
    label: "Test WAPE",
    numeric: true,
    sortValue: (r) => winnerAttempt(r)?.diagnostics.test_wape ?? null,
    format: (r) => fmt(winnerAttempt(r)?.diagnostics.test_wape, 3),
  },
  {
    key: "n_train",
    label: "N train",
    numeric: true,
    format: (r) => `${r.n_train}`,
  },
  {
    key: "sign_ok",
    label: "Sign OK",
    align: "center",
    sortValue: (r) => (winnerAttempt(r)?.sign_ok ? 1 : 0),
    format: (r) => {
      const ok = winnerAttempt(r)?.sign_ok;
      return ok ? (
        <span className="rounded border border-emerald-500/40 bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">
          ✓
        </span>
      ) : (
        <span className="rounded border border-rose-500/40 bg-rose-500/15 px-1.5 py-0.5 text-[10px] text-rose-300">
          ✗
        </span>
      );
    },
  },
  {
    key: "retry",
    label: "Retry",
    align: "center",
    sortValue: (r) => (r.sign_retry_fired ? 1 : 0),
    format: (r) => (r.sign_retry_fired ? "yes" : "—"),
  },
];

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
              <tr
                key={a.model}
                className={isWinner ? "bg-emerald-500/5 text-slate-200" : "text-slate-400"}
              >
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

export interface CandidatesTableProps {
  rows: CandidatesRow[];
  selectedPpg: string | null;
  onSelectPpg: (ppg: string) => void;
}

export function CandidatesTable({ rows, selectedPpg, onSelectPpg }: CandidatesTableProps) {
  return (
    <ResultsTable
      rows={rows}
      columns={COLUMNS}
      rowKey={(r) => r.ppg_id}
      empty="No fits to display."
      defaultSort={{ key: "test_wape", dir: "asc" }}
      highlightKey={selectedPpg}
      onRowClick={(r) => onSelectPpg(r.ppg_id)}
      expandable={{ render: (r) => <AttemptsList row={r} /> }}
    />
  );
}
