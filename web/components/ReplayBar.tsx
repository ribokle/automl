"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { fmtClock } from "@/lib/agent-meta";
import type { AgentName, RunEvent } from "@/lib/types";

interface Props {
  events: RunEvent[];
  /** ISO timestamp of "frozen" position. null = live tail. */
  scrubTs: string | null;
  onScrub: (ts: string | null) => void;
}

const PHASE_COLOR: Record<string, string> = {
  agent_started: "bg-amber-400",
  agent_finished: "bg-emerald-400",
  agent_failed: "bg-rose-500",
  approval_required: "bg-purple-400",
  approval_resolved: "bg-emerald-500",
  agent_rerunning: "bg-amber-500",
  tool_called: "bg-slate-500",
  run_started: "bg-sky-400",
  run_finished: "bg-emerald-500",
};

function fmtElapsed(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s - m * 60}s`;
}

export function ReplayBar({ events, scrubTs, onScrub }: Props) {
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const ordered = useMemo(
    () =>
      [...events]
        .map((e) => ({ ev: e, t: Date.parse(e.ts) }))
        .filter((x) => Number.isFinite(x.t))
        .sort((a, b) => a.t - b.t),
    [events],
  );

  const t0 = ordered[0]?.t ?? null;
  const tLast = ordered[ordered.length - 1]?.t ?? null;

  // While playing, advance scrubTs at a fixed step until we reach tLast.
  useEffect(() => {
    if (!playing || t0 == null || tLast == null) return;
    if (timerRef.current) clearInterval(timerRef.current);
    const totalMs = tLast - t0;
    const step = Math.max(150, Math.floor(totalMs / 80)); // ~80 frames across the run
    timerRef.current = setInterval(() => {
      const currentMs = scrubTs ? Date.parse(scrubTs) : t0;
      const next = Math.min(tLast, currentMs + step);
      onScrub(new Date(next).toISOString());
      if (next >= tLast) {
        setPlaying(false);
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }, 60);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing, t0, tLast, scrubTs, onScrub]);

  // Keyboard shortcuts: space toggles play/pause, ←/→ step one event tick.
  useEffect(() => {
    if (t0 == null || tLast == null || ordered.length < 2) return;
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        target?.isContentEditable
      ) {
        return;
      }
      if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        if (playing) {
          setPlaying(false);
          return;
        }
        if (!scrubTs || Date.parse(scrubTs) >= tLast) {
          onScrub(new Date(t0!).toISOString());
        }
        setPlaying(true);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        setPlaying(false);
        const currentMs = scrubTs ? Date.parse(scrubTs) : tLast!;
        if (e.key === "ArrowLeft") {
          const prev = ordered
            .filter(({ t }) => t < currentMs - 1)
            .reduce<number | null>((acc, x) => (acc === null || x.t > acc ? x.t : acc), null);
          if (prev !== null) onScrub(new Date(prev).toISOString());
        } else {
          const next = ordered.find(({ t }) => t > currentMs + 1)?.t ?? null;
          // Clamp to tLast rather than jumping to live — user must click "live" explicitly.
          if (next !== null) onScrub(new Date(Math.min(next, tLast!)).toISOString());
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [playing, scrubTs, ordered, t0, tLast, onScrub]);

  if (ordered.length < 2 || t0 == null || tLast == null) return null;

  const totalMs = tLast - t0;
  const currentMs = scrubTs ? Date.parse(scrubTs) : tLast;
  const isLive = scrubTs == null;
  const pct = totalMs > 0 ? ((currentMs - t0) / totalMs) * 100 : 100;

  // Compute event ticks for the rail.
  const ticks = ordered.map(({ ev, t }) => {
    const left = totalMs > 0 ? ((t - t0) / totalMs) * 100 : 0;
    return {
      left,
      color: PHASE_COLOR[ev.type] ?? "bg-slate-600",
      title: `${ev.agent ?? ""} · ${ev.type}`,
    };
  });

  // Agent that was most recently active at currentMs (for the "now showing" label).
  const lastBefore = ordered
    .filter(({ t }) => t <= currentMs)
    .reverse()
    .find(({ ev }) => ev.agent && ev.type !== "tool_called");
  const focusAgent = (lastBefore?.ev.agent as AgentName | undefined) ?? null;
  const focusEv = lastBefore?.ev.type ?? null;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wider text-slate-500">Replay</span>
          {isLive ? (
            <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-emerald-300">
              ● live
            </span>
          ) : (
            <span className="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-amber-300">
              frozen at {fmtClock(scrubTs)}
            </span>
          )}
          {focusAgent && (
            <span className="font-mono text-[11px] text-slate-300">
              {focusAgent} · {focusEv?.replace(/_/g, " ")}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => {
              if (playing) {
                setPlaying(false);
                return;
              }
              // If at end, restart from t0; else continue from current.
              if (!scrubTs || Date.parse(scrubTs) >= tLast) {
                onScrub(new Date(t0).toISOString());
              }
              setPlaying(true);
            }}
            className="rounded border border-slate-700 bg-slate-800 px-2 py-0.5 text-[11px] text-slate-200 hover:bg-slate-700"
            aria-label={playing ? "pause replay" : "play replay"}
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <button
            type="button"
            onClick={() => {
              setPlaying(false);
              onScrub(null);
            }}
            className="rounded border border-slate-700 bg-slate-800 px-2 py-0.5 text-[11px] text-slate-200 hover:bg-slate-700"
            aria-label="return to live"
          >
            live
          </button>
          <span
            className="hidden font-mono text-[10px] text-slate-500 sm:inline"
            title="space = play/pause · ← / → = step one event"
          >
            <kbd className="rounded border border-slate-700 px-1">␣</kbd>{" "}
            <kbd className="rounded border border-slate-700 px-1">←</kbd>{" "}
            <kbd className="rounded border border-slate-700 px-1">→</kbd>
          </span>
        </div>
      </div>

      <div className="relative h-7">
        <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded bg-slate-800" />
        {ticks.map((t, i) => (
          <span
            key={i}
            title={t.title}
            className={`absolute top-1/2 h-2 w-px -translate-y-1/2 ${t.color}`}
            style={{ left: `${t.left}%` }}
          />
        ))}
        <div
          className="absolute top-1/2 h-1 -translate-y-1/2 rounded bg-emerald-500/40"
          style={{ left: 0, width: `${pct}%` }}
        />
        <input
          type="range"
          min={0}
          max={totalMs > 0 ? totalMs : 1}
          step={1}
          value={Math.max(0, currentMs - t0)}
          onChange={(e) => {
            setPlaying(false);
            const offsetMs = Number(e.target.value);
            const next = t0 + offsetMs;
            // Snap to "live" if user drags to the very end.
            if (next >= tLast) onScrub(null);
            else onScrub(new Date(next).toISOString());
          }}
          className="absolute inset-0 h-full w-full cursor-pointer appearance-none bg-transparent [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-emerald-400 [&::-webkit-slider-thumb]:shadow"
          aria-label="replay position"
        />
      </div>

      <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-500">
        <span>t=0 · {fmtClock(new Date(t0).toISOString())}</span>
        <span>{fmtElapsed(currentMs - t0)} elapsed</span>
        <span>
          {fmtClock(new Date(tLast).toISOString())} · {fmtElapsed(totalMs)} total
        </span>
      </div>
    </div>
  );
}
