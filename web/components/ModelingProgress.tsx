"use client";

/**
 * Streaming progress bar for the modelling agent.
 *
 * The modelling agent emits one ``fit_candidates`` event per successful
 * cell fit and one ``fit_skipped`` event per cell that didn't pass the
 * pre-fit gate. The ``load_inputs`` event carries ``total_cells``, the
 * denominator both modes converge on (one cell per PPG at chain grain,
 * one cell per (store, PPG) at the Hoch grain).
 *
 * This component is a thin client-side counter over those events. While
 * modelling is running it shows a moving bar plus the most recently
 * fit cell ID; when modelling is ``done`` the bar pins to 100% and the
 * panel collapses to a one-line summary that downstream agents'
 * sections can sit comfortably below.
 */

import { useMemo } from "react";
import type { AgentState, RunEvent } from "@/lib/types";

interface Props {
  events: RunEvent[];
  agentState?: AgentState;
}

interface Counts {
  fit: number;
  skipped: number;
  total: number;
  lastPpg?: string;
  lastGrainUnit?: string | null;
  lastWinner?: string;
}

function collectCounts(events: RunEvent[]): Counts {
  let fit = 0;
  let skipped = 0;
  let total = 0;
  let lastPpg: string | undefined;
  let lastGrainUnit: string | null | undefined;
  let lastWinner: string | undefined;
  for (const ev of events) {
    if (ev.agent !== "modeling" || ev.type !== "tool_called") continue;
    if (ev.tool === "load_inputs" && typeof ev.total_cells === "number") {
      total = ev.total_cells;
    } else if (ev.tool === "fit_candidates") {
      fit += 1;
      lastPpg = ev.ppg_id;
      lastGrainUnit = ev.grain_unit ?? null;
      lastWinner = ev.winner;
    } else if (ev.tool === "fit_skipped") {
      skipped += 1;
      lastPpg = ev.ppg_id;
      lastGrainUnit = ev.grain_unit ?? null;
      lastWinner = "skipped";
    }
  }
  return { fit, skipped, total, lastPpg, lastGrainUnit, lastWinner };
}

function MiniDots({ done, total }: { done: number; total: number }) {
  // Up to 50 dots; clamp so the visual stays compact on big runs.
  const cap = 50;
  const cells = Math.min(total || done, cap);
  if (cells === 0) return null;
  const filled = Math.round((done / Math.max(total, done, 1)) * cells);
  return (
    <div className="flex flex-wrap gap-0.5" aria-hidden="true">
      {Array.from({ length: cells }).map((_, i) => (
        <span
          key={i}
          className={`block h-1.5 w-1.5 rounded-sm ${
            i < filled ? "bg-emerald-400/80" : "bg-slate-700/60"
          }`}
        />
      ))}
    </div>
  );
}

export function ModelingProgress({ events, agentState }: Props) {
  const counts = useMemo(() => collectCounts(events), [events]);
  const status = agentState?.status ?? "pending";

  // Before any modelling event arrives there's nothing useful to show.
  const seenAny = counts.fit + counts.skipped > 0 || counts.total > 0;
  if (!seenAny) return null;

  const done = counts.fit + counts.skipped;
  const denom = Math.max(counts.total, done, 1);
  const pct = Math.min(100, Math.round((done / denom) * 100));
  const isRunning = status === "running";
  const isDone = status === "done";
  const isFailed = status === "failed";

  const barColor = isFailed
    ? "bg-rose-500/80"
    : isDone
      ? "bg-emerald-500/80"
      : "bg-sky-400/80";

  const liveLabel = counts.lastPpg
    ? `· last: ${counts.lastPpg}${counts.lastGrainUnit ? `@${counts.lastGrainUnit}` : ""}${
        counts.lastWinner ? ` (${counts.lastWinner})` : ""
      }`
    : "";

  return (
    <div
      className="rounded border border-slate-800 bg-slate-950/60 px-3 py-2 text-[11px] text-slate-300"
      role="status"
      aria-live={isRunning ? "polite" : "off"}
    >
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="font-medium text-slate-200">
          Modelling progress
          {isRunning && (
            <span className="ml-2 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-sky-400" />
          )}
        </span>
        <span className="font-mono tabular-nums text-slate-400">
          {done.toLocaleString()} / {counts.total ? counts.total.toLocaleString() : "?"} cells
          {counts.skipped > 0 && (
            <span className="ml-1 text-amber-300">({counts.skipped} skipped)</span>
          )}
          <span className="ml-2 text-slate-500">· {pct}%</span>
        </span>
      </div>

      <div className="relative mb-1.5 h-2 overflow-hidden rounded bg-slate-800/80">
        <div
          className={`absolute inset-y-0 left-0 ${barColor} transition-[width] duration-200`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Compact dot-grid: nice spatial sense of how many cells are still
          outstanding, capped so wide runs (~2,300 cells on toothpaste at
          store-grain) stay readable. */}
      <MiniDots done={done} total={denom} />

      {liveLabel && (
        <div className="mt-1 text-[10px] text-slate-500">{liveLabel}</div>
      )}
    </div>
  );
}
