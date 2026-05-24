"use client";

/**
 * Modelling-grain selector for the post-ingestion approval gate.
 *
 * Shows a 3×2 grid of grain options (product axis: PPG / Category /
 * Brand × spatial axis: chain / store) with expected cell counts +
 * recommendation badges + greyed-out unavailable cells. Below the
 * primary radio, a multi-select chip row lets the operator queue
 * comparison grains for the modelling-only fan-out.
 *
 * Mounts inside the ``ppg_mapping`` approval panel in AgentCard.
 * Approve button calls ``approveAgent(runId, "ppg_mapping",
 * { modelling_grain, comparison_grains })`` so the chosen config
 * flows into ``run.options`` via the orchestrator's approve-payload
 * machinery (see ``core/orchestrator/runner.py:_wait_for_gate``).
 */

import { useEffect, useMemo, useState } from "react";

export interface GrainOption {
  id: string;
  label: string;
  spatial_axis: "chain" | "store";
  product_axis: "ppg" | "category" | "brand";
  expected_cells: number;
  expected_rows_per_cell: number;
  available: boolean;
  recommended: boolean;
  reason: string;
}

export interface GrainOptionsBlob {
  shape: {
    n_stores: number;
    n_ppgs: number;
    n_categories: number;
    n_brands: number;
    n_weeks: number;
  } | null;
  options: GrainOption[];
}

interface Props {
  options: GrainOptionsBlob;
  initialPrimary?: string;
  onSubmit: (payload: {
    modelling_grain: string;
    comparison_grains: string[];
    comparison_agents: string[];
  }) => Promise<void>;
  submitLabel?: string;
  disabled?: boolean;
}

// Operator-facing fan-out depth stages. Must match
// ``_COMPARISON_AGENT_CHOICES`` in ``api/routes/approvals.py`` and the
// canonical order in ``core/orchestrator/runner.py``. The list is
// ordered: picking a stage implicitly includes every stage above it.
const FANOUT_STAGES: Array<{ id: string; label: string; description: string }> = [
  { id: "modeling", label: "Modelling", description: "elasticities + fit diagnostics" },
  { id: "decomposition", label: "Decomposition", description: "due-to driver split" },
  { id: "simulation", label: "Simulation", description: "price × promo grid" },
  { id: "optimization", label: "Optimisation", description: "recommended prices" },
  { id: "validation", label: "Validation", description: "rolling-CV verdicts" },
  { id: "insights", label: "Insights", description: "executive narrative (LLM)" },
];

const PRODUCT_ROWS: Array<{ id: "ppg" | "category" | "brand"; label: string; description: string }> = [
  { id: "ppg", label: "PPG", description: "auto-clustered price-pack groups" },
  { id: "category", label: "Category", description: "category labels from the panel" },
  { id: "brand", label: "Brand", description: "brand labels from the panel" },
];

const SPATIAL_COLS: Array<{ id: "chain" | "store"; label: string; description: string }> = [
  { id: "chain", label: "Chain", description: "collapses stores" },
  { id: "store", label: "Store", description: "one model per store" },
];

function fmtCells(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return `${n}`;
}

export function GrainSelector({ options, initialPrimary, onSubmit, submitLabel = "Approve & Configure", disabled }: Props) {
  const optsByKey = useMemo(() => {
    const map = new Map<string, GrainOption>();
    for (const o of options.options) {
      map.set(`${o.spatial_axis}:${o.product_axis}`, o);
    }
    return map;
  }, [options]);

  const recommendation = useMemo(
    () => options.options.find((o) => o.recommended && o.available)?.id,
    [options],
  );
  const defaultPrimary = initialPrimary
    || recommendation
    || options.options.find((o) => o.available)?.id
    || "ppg_week";

  const [primary, setPrimary] = useState<string>(defaultPrimary);
  const [comparisons, setComparisons] = useState<Set<string>>(new Set());
  const [fanout, setFanout] = useState<Set<string>>(
    () => new Set(FANOUT_STAGES.map((s) => s.id)),
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPrimary(defaultPrimary);
  }, [defaultPrimary]);

  function toggleFanout(stageId: string) {
    // Cascade rules: unchecking a stage removes every stage AFTER it
    // (data dependency); checking a stage adds every stage BEFORE it.
    // The operator never sees a half-valid configuration.
    setFanout((prev) => {
      const idx = FANOUT_STAGES.findIndex((s) => s.id === stageId);
      if (idx < 0) return prev;
      const next = new Set(prev);
      if (next.has(stageId)) {
        for (let i = idx; i < FANOUT_STAGES.length; i++) next.delete(FANOUT_STAGES[i].id);
      } else {
        for (let i = 0; i <= idx; i++) next.add(FANOUT_STAGES[i].id);
      }
      return next;
    });
  }

  function toggleComparison(id: string) {
    setComparisons((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      // Comparison can't equal primary; drop it if the operator picks it as primary later.
      next.delete(primary);
      return next;
    });
  }

  function pickPrimary(id: string) {
    setPrimary(id);
    setComparisons((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }

  async function handleSubmit() {
    setError(null);
    setSubmitting(true);
    try {
      await onSubmit({
        modelling_grain: primary,
        comparison_grains: Array.from(comparisons),
        comparison_agents: FANOUT_STAGES.map((s) => s.id).filter((id) => fanout.has(id)),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-3 rounded border border-sky-500/30 bg-sky-500/[0.04] p-3 text-[11px] text-slate-200">
      <div>
        <div className="flex items-baseline justify-between">
          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-sky-200">
            Modelling grain
          </h4>
          {options.shape && (
            <span className="text-[10px] text-slate-500">
              {options.shape.n_stores} stores · {options.shape.n_ppgs} PPGs ·
              {" "}{options.shape.n_categories} categories ·
              {" "}{options.shape.n_brands} brands ·
              {" "}{options.shape.n_weeks} weeks
            </span>
          )}
        </div>
        <p className="mt-0.5 text-[10px] text-slate-500">
          Pick the cell the modelling agent fits one demand model per. Stars mark a sensible default.
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="text-[11px]">
          <thead className="text-[9.5px] uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-2 py-1 text-left"></th>
              {SPATIAL_COLS.map((col) => (
                <th key={col.id} className="px-2 py-1 text-left">
                  <div>{col.label} × week</div>
                  <div className="text-[8.5px] font-normal normal-case text-slate-600">{col.description}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {PRODUCT_ROWS.map((row) => (
              <tr key={row.id} className="border-t border-slate-800/60">
                <th className="px-2 py-2 text-left font-medium text-slate-300">
                  <div>{row.label}</div>
                  <div className="text-[9px] font-normal text-slate-600">{row.description}</div>
                </th>
                {SPATIAL_COLS.map((col) => {
                  const opt = optsByKey.get(`${col.id}:${row.id}`);
                  if (!opt) return <td key={col.id} className="px-2 py-2" />;
                  const isPrimary = primary === opt.id;
                  const isCompare = comparisons.has(opt.id);
                  const disabledCell = !opt.available || disabled || submitting;
                  return (
                    <td key={col.id} className="px-2 py-1.5">
                      <label
                        title={opt.reason}
                        className={`flex cursor-pointer items-start gap-2 rounded border px-2 py-1.5 text-[11px] ${
                          disabledCell
                            ? "cursor-not-allowed border-slate-800 bg-slate-900/40 text-slate-600"
                            : isPrimary
                              ? "border-sky-400/70 bg-sky-500/15 text-sky-100"
                              : "border-slate-700 bg-slate-900/60 hover:border-slate-500"
                        }`}
                      >
                        <input
                          type="radio"
                          className="mt-0.5 accent-sky-400"
                          name="primary-grain"
                          checked={isPrimary}
                          disabled={disabledCell}
                          onChange={() => !disabledCell && pickPrimary(opt.id)}
                        />
                        <div className="leading-tight">
                          <div className="flex items-center gap-1.5 font-mono">
                            {fmtCells(opt.expected_cells)} cells
                            {opt.recommended && opt.available && (
                              <span className="text-amber-300" aria-label="recommended">★</span>
                            )}
                          </div>
                          <div className="text-[9.5px] text-slate-500">
                            {opt.available ? "" : opt.reason}
                          </div>
                          {opt.available && !isPrimary && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                toggleComparison(opt.id);
                              }}
                              disabled={disabledCell}
                              className={`mt-1 rounded border px-1.5 py-0.5 text-[9.5px] uppercase tracking-wider ${
                                isCompare
                                  ? "border-emerald-400/60 bg-emerald-500/15 text-emerald-200"
                                  : "border-slate-700 text-slate-400 hover:border-slate-500"
                              }`}
                            >
                              {isCompare ? "✓ comparing" : "+ compare"}
                            </button>
                          )}
                        </div>
                      </label>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {comparisons.size > 0 && (
        <div className="space-y-2 rounded border border-emerald-500/30 bg-emerald-500/[0.06] px-2 py-2 text-[10px] text-emerald-100">
          <div>
            <span className="font-semibold uppercase tracking-wider">Comparison grains</span>
            {": "}
            {Array.from(comparisons).join(", ")}
          </div>
          <div>
            <div className="mb-1 font-semibold uppercase tracking-wider">Fan-out depth</div>
            <div className="text-[9.5px] text-emerald-200/70">
              Stages re-run per comparison grain. Unchecking a stage drops every stage after it.
            </div>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {FANOUT_STAGES.map((stage) => {
                const checked = fanout.has(stage.id);
                return (
                  <label
                    key={stage.id}
                    title={stage.description}
                    className={`flex cursor-pointer items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] ${
                      checked
                        ? "border-emerald-400/60 bg-emerald-500/15 text-emerald-100"
                        : "border-slate-700 bg-slate-900/40 text-slate-400"
                    }`}
                  >
                    <input
                      type="checkbox"
                      className="accent-emerald-400"
                      checked={checked}
                      onChange={() => toggleFanout(stage.id)}
                    />
                    {stage.label}
                  </label>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded border border-rose-500/40 bg-rose-500/10 px-2 py-1.5 text-rose-200">
          {error}
        </div>
      )}

      <div className="flex justify-end">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={submitting || disabled}
          className="rounded border border-sky-400/60 bg-sky-500/20 px-3 py-1 text-[11px] text-sky-100 hover:bg-sky-500/30 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitting ? "Submitting…" : submitLabel}
        </button>
      </div>
    </div>
  );
}
