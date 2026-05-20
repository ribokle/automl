import type { AgentStatus } from "./types";

export const STATUS_STYLE: Record<
  AgentStatus | "idle",
  { dot: string; pill: string; ring: string }
> = {
  idle: {
    dot: "bg-slate-700",
    pill: "bg-slate-800 text-slate-400 border-slate-700",
    ring: "ring-slate-800",
  },
  pending: {
    dot: "bg-slate-700",
    pill: "bg-slate-800 text-slate-400 border-slate-700",
    ring: "ring-slate-800",
  },
  running: {
    dot: "bg-amber-400 animate-pulse",
    pill: "bg-amber-500/15 text-amber-300 border-amber-500/40",
    ring: "ring-amber-500/40",
  },
  awaiting_approval: {
    dot: "bg-purple-400 animate-pulse",
    pill: "bg-purple-500/15 text-purple-300 border-purple-500/40",
    ring: "ring-purple-500/40",
  },
  done: {
    dot: "bg-emerald-400",
    pill: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
    ring: "ring-emerald-500/30",
  },
  failed: {
    dot: "bg-rose-500",
    pill: "bg-rose-500/15 text-rose-300 border-rose-500/40",
    ring: "ring-rose-500/40",
  },
  skipped: {
    dot: "bg-slate-600",
    pill: "bg-slate-800/60 text-slate-500 border-slate-700",
    ring: "ring-slate-800",
  },
};

// RunSidebar uses a slightly different status vocabulary ("completed" rather
// than "done", plus a "pending" fallback for unknown states); keep that
// mapping derived from STATUS_STYLE so the dot colour can never disagree.
export const STATUS_DOT: Record<string, string> = {
  completed: STATUS_STYLE.done.dot,
  running: STATUS_STYLE.running.dot,
  awaiting_approval: STATUS_STYLE.awaiting_approval.dot,
  failed: STATUS_STYLE.failed.dot,
  pending: STATUS_STYLE.pending.dot,
};

export const PHASE_COLOR: Record<string, string> = {
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
