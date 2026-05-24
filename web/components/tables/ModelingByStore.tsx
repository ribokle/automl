"use client";

/**
 * Per-store elasticity drilldown for runs at the Hoch-style grain
 * (`store_ppg_week` / `store_category_week`).
 *
 * At those grains the modelling agent fits one model per (store, PPG)
 * cell. The dashboard's existing one-row-per-PPG table can't represent
 * that — this component renders:
 *
 *   PPG_A · pooled ε (inverse-variance) over N stores         ▸ expand
 *     └─ store_001 · loglog · ε=-1.31  R²=0.78  WAPE=24%      ▸ expand
 *           └─ attempts table (loglog / semilog / lightgbm)
 *     └─ store_002 · loglog · ε=-1.42  R²=0.81  WAPE=22%
 *     ...
 *
 * For the chain grain (default), `ModelingVisuals` keeps using
 * `CandidatesTable` — this component is mounted only when at least one
 * modelling row carries a `grain_unit` other than `null` / `"chain"`.
 */

import { ResultsTable, type ColumnDef } from "./ResultsTable";
import type { CandidatesRow, CandidateAttempt } from "./CandidatesTable";

export interface PooledRow {
  ppg_id: string;
  grain_unit: string; // "pooled"
  model: string; // "store_inverse_variance_pool"
  own_elasticity: number;
  std_err: number;
  r_squared: number | null;
  test_wape: number | null;
  n_obs: number;
  n_stores_pooled: number;
  sign_ok: boolean;
}

/** A single (store, PPG) modelling row: same shape as a chain-grain
 *  CandidatesRow, but with `grain_unit` populated. */
export type CellRow = CandidatesRow & { grain_unit: string };

const MODEL_LABEL: Record<string, string> = {
  loglog_ols: "log-log OLS",
  semilog_ols: "semi-log OLS",
  lightgbm: "LightGBM",
  store_inverse_variance_pool: "pooled (IV)",
};

function fmt(n: number | undefined | null, digits = 3): string {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  if (!Number.isFinite(n)) return "∞";
  return n.toFixed(digits);
}

function winnerAttempt(row: CandidatesRow): CandidateAttempt | undefined {
  return row.attempts.find((a) => a.model === row.winner_model);
}

/* ------- Inner table: attempts under one (store, PPG) cell ------- */

const ATTEMPT_COLS: ColumnDef<CandidateAttempt & { isWinner: boolean }>[] = [
  {
    key: "model",
    label: "Model",
    format: (a) => (
      <span className="font-mono">
        {MODEL_LABEL[a.model] ?? a.model}
        {a.isWinner && (
          <span className="ml-2 rounded border border-emerald-500/40 px-1 text-[9px] text-emerald-300">
            winner
          </span>
        )}
      </span>
    ),
  },
  { key: "own_elasticity", label: "Elasticity", numeric: true, format: (a) => fmt(a.own_elasticity, 2) },
  { key: "std_err", label: "Std err", numeric: true, format: (a) => fmt(a.std_err, 3) },
  { key: "r_squared", label: "R²", numeric: true, format: (a) => fmt(a.r_squared, 2) },
  {
    key: "test_wape",
    label: "Test WAPE",
    numeric: true,
    sortValue: (a) => a.diagnostics.test_wape ?? null,
    format: (a) => fmt(a.diagnostics.test_wape, 3),
  },
  {
    key: "sign_ok",
    label: "Sign",
    align: "center",
    sortValue: (a) => (a.sign_ok ? 1 : 0),
    format: (a) =>
      a.sign_ok ? (
        <span className="text-emerald-300">✓</span>
      ) : (
        <span className="text-rose-300">✗</span>
      ),
  },
];

function AttemptsForCell({ cell }: { cell: CellRow }) {
  const enriched = cell.attempts.map((a) => ({ ...a, isWinner: a.model === cell.winner_model }));
  return (
    <div className="space-y-1">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">
        Candidates · {cell.ppg_id} @ {cell.grain_unit}
      </div>
      <ResultsTable
        rows={enriched}
        columns={ATTEMPT_COLS}
        rowKey={(a) => a.model}
        highlightKey={cell.winner_model}
        stickyFirst={false}
        empty="No attempts recorded."
      />
    </div>
  );
}

/* ------- Middle table: store cells under one PPG ------- */

const CELL_COLS: ColumnDef<CellRow>[] = [
  { key: "grain_unit", label: "Store", format: (c) => <span className="font-mono">{c.grain_unit}</span> },
  {
    key: "winner_model",
    label: "Winner",
    format: (c) => MODEL_LABEL[c.winner_model] ?? c.winner_model,
  },
  {
    key: "elasticity",
    label: "Elasticity",
    numeric: true,
    sortValue: (c) => winnerAttempt(c)?.own_elasticity ?? null,
    format: (c) => fmt(winnerAttempt(c)?.own_elasticity, 2),
  },
  {
    key: "r_squared",
    label: "R²",
    numeric: true,
    sortValue: (c) => winnerAttempt(c)?.r_squared ?? null,
    format: (c) => fmt(winnerAttempt(c)?.r_squared, 2),
  },
  {
    key: "test_wape",
    label: "Test WAPE",
    numeric: true,
    sortValue: (c) => winnerAttempt(c)?.diagnostics.test_wape ?? null,
    format: (c) => fmt(winnerAttempt(c)?.diagnostics.test_wape, 3),
  },
  {
    key: "n_train",
    label: "N train",
    numeric: true,
    format: (c) => `${c.n_train}`,
  },
  {
    key: "sign_ok",
    label: "Sign",
    align: "center",
    sortValue: (c) => (winnerAttempt(c)?.sign_ok ? 1 : 0),
    format: (c) =>
      winnerAttempt(c)?.sign_ok ? (
        <span className="text-emerald-300">✓</span>
      ) : (
        <span className="text-rose-300">✗</span>
      ),
  },
];

function CellList({ ppgId, cells }: { ppgId: string; cells: CellRow[] }) {
  return (
    <div className="space-y-1">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">
        Per-store cells · {ppgId} · {cells.length} stores
      </div>
      <ResultsTable
        rows={cells}
        columns={CELL_COLS}
        rowKey={(c) => `${c.ppg_id}@${c.grain_unit}`}
        defaultSort={{ key: "test_wape", dir: "asc" }}
        stickyFirst={false}
        empty="No store cells."
        expandable={{ render: (c) => <AttemptsForCell cell={c} /> }}
      />
    </div>
  );
}

/* ------- Outer table: one row per PPG (the pooled view) ------- */

interface OuterRow {
  ppg_id: string;
  pooled: PooledRow | null;
  cells: CellRow[];
}

const OUTER_COLS: ColumnDef<OuterRow>[] = [
  { key: "ppg_id", label: "PPG" },
  {
    key: "pooled_elasticity",
    label: "Pooled ε (IV)",
    numeric: true,
    sortValue: (r) => r.pooled?.own_elasticity ?? null,
    format: (r) => fmt(r.pooled?.own_elasticity, 2),
  },
  {
    key: "pooled_se",
    label: "Pooled SE",
    numeric: true,
    sortValue: (r) => r.pooled?.std_err ?? null,
    format: (r) => fmt(r.pooled?.std_err, 3),
  },
  {
    key: "n_stores",
    label: "Stores",
    numeric: true,
    sortValue: (r) => r.pooled?.n_stores_pooled ?? r.cells.length,
    format: (r) => `${r.pooled?.n_stores_pooled ?? r.cells.length}`,
  },
  {
    key: "pooled_sign",
    label: "Sign",
    align: "center",
    sortValue: (r) => (r.pooled?.sign_ok ? 1 : 0),
    format: (r) =>
      r.pooled?.sign_ok ? (
        <span className="rounded border border-emerald-500/40 bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">
          ✓
        </span>
      ) : (
        <span className="rounded border border-rose-500/40 bg-rose-500/15 px-1.5 py-0.5 text-[10px] text-rose-300">
          ✗
        </span>
      ),
  },
];

export interface ModelingByStoreProps {
  /** Modelling rows that carry a non-pooled `grain_unit` (per-store cells). */
  cells: CellRow[];
  /** Inverse-variance-pooled rows from `elasticity_per_ppg_pooled.json`. */
  pooled: PooledRow[];
  selectedPpg: string | null;
  onSelectPpg: (ppg: string) => void;
}

export function ModelingByStore({
  cells,
  pooled,
  selectedPpg,
  onSelectPpg,
}: ModelingByStoreProps) {
  const cellsByPpg = new Map<string, CellRow[]>();
  for (const c of cells) {
    const list = cellsByPpg.get(c.ppg_id) ?? [];
    list.push(c);
    cellsByPpg.set(c.ppg_id, list);
  }
  const pooledByPpg = new Map<string, PooledRow>(pooled.map((p) => [p.ppg_id, p]));
  const ppgIds = Array.from(new Set([...cellsByPpg.keys(), ...pooledByPpg.keys()])).sort();
  const rows: OuterRow[] = ppgIds.map((ppgId) => ({
    ppg_id: ppgId,
    pooled: pooledByPpg.get(ppgId) ?? null,
    cells: cellsByPpg.get(ppgId) ?? [],
  }));

  return (
    <ResultsTable
      rows={rows}
      columns={OUTER_COLS}
      rowKey={(r) => r.ppg_id}
      empty="No store cells to display."
      defaultSort={{ key: "pooled_elasticity", dir: "desc" }}
      highlightKey={selectedPpg}
      onRowClick={(r) => onSelectPpg(r.ppg_id)}
      expandable={{ render: (r) => <CellList ppgId={r.ppg_id} cells={r.cells} /> }}
    />
  );
}
