"use client";

/**
 * Per-card chip row for switching between the primary grain and any
 * comparison grains the operator queued at the ppg_mapping gate.
 *
 * Renders nothing when no comparison grains are configured, so cards
 * that adopt this look identical for single-grain runs. The primary
 * grain is marked with a star.
 */

import type { GrainState } from "@/lib/useGrainState";

interface Props {
  state: GrainState;
  progress?: { grain: string; agentIndex: number; totalAgents: number } | null;
}

export function GrainChipRow({ state, progress }: Props) {
  if (state.comparisons.length === 0) return null;
  const grains = [state.primary, ...state.comparisons];
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
      <span className="uppercase tracking-wider text-slate-500">Grain:</span>
      {grains.map((g) => {
        const isPrimary = g === state.primary;
        const isActive = g === state.selected;
        const isProgressing = progress && progress.grain === g;
        return (
          <button
            key={g}
            type="button"
            onClick={() => state.setSelected(g)}
            className={`rounded border px-2 py-0.5 font-mono transition ${
              isActive
                ? "border-sky-400/70 bg-sky-500/15 text-sky-100"
                : "border-slate-700 text-slate-400 hover:border-slate-500"
            }`}
          >
            {g}
            {isPrimary && (
              <span className="ml-1 text-amber-300" title="primary grain">
                ★
              </span>
            )}
            {isProgressing && (
              <span className="ml-1 text-emerald-300" title="comparison running">
                ({progress.agentIndex + 1}/{progress.totalAgents})
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
