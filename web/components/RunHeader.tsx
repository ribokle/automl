"use client";

import { useState } from "react";
import { AGENT_ORDER, type AgentName, type AgentState } from "@/lib/types";
import { formatDisplayName, formatDuration, STATUS_STYLE } from "@/lib/agent-meta";
import { approveAgent, rejectAgent } from "@/lib/api";

interface Props {
  runId: string;
  runStatus: string;
  agents: Record<AgentName, AgentState> | null;
  startedAt: string | null;
}

function elapsed(start: string | null, agents: Record<AgentName, AgentState> | null): string {
  if (!agents) return formatDuration(start, null) ?? "—";
  const finishedAll = AGENT_ORDER.every((a) => agents[a]?.status === "done" || agents[a]?.status === "skipped");
  const anyAwaiting = AGENT_ORDER.some((a) => agents[a]?.status === "awaiting_approval");
  const lastEnd = AGENT_ORDER.map((a) => agents[a]?.finished_at).filter(Boolean).sort().pop() as string | undefined;
  // When paused at a gate, cap at when the gated agent finished — don't count wait time.
  if (anyAwaiting && lastEnd) return formatDuration(start, lastEnd) ?? "—";
  return formatDuration(start, finishedAll ? lastEnd ?? null : null) ?? "—";
}

export function RunHeader({ runId, runStatus, agents, startedAt }: Props) {
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const done = agents
    ? AGENT_ORDER.filter((a) => agents[a]?.status === "done" || agents[a]?.status === "skipped").length
    : 0;
  const running = agents ? AGENT_ORDER.find((a) => agents[a]?.status === "running") : undefined;
  const awaiting = agents ? AGENT_ORDER.find((a) => agents[a]?.status === "awaiting_approval") : undefined;
  const total = AGENT_ORDER.length;
  const pct = Math.round((done / total) * 100);

  const statusKey = (
    runStatus === "completed"
      ? "done"
      : runStatus === "failed"
        ? "failed"
        : awaiting
          ? "awaiting_approval"
          : "running"
  ) as keyof typeof STATUS_STYLE;
  const style = STATUS_STYLE[statusKey];

  const currentLabel = awaiting
    ? `awaiting approval · ${formatDisplayName(awaiting)}`
    : running
      ? `running · ${formatDisplayName(running)}`
      : runStatus;

  function jumpToAwaiting() {
    if (!awaiting) return;
    const target =
      document.getElementById(`approval-${awaiting}`) ??
      document.getElementById(`agent-${awaiting}`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  async function handleApprove() {
    if (!awaiting || busy) return;
    setBusy(true);
    setActionError(null);
    try {
      await approveAgent(runId, awaiting);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleReject() {
    if (!awaiting || busy) return;
    setBusy(true);
    setActionError(null);
    try {
      await rejectAgent(runId, awaiting);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-wider text-slate-500">Run</p>
            <h1 className="font-mono text-lg text-slate-200">{runId}</h1>
          </div>
          <div className="flex items-center gap-3">
            <span className={`rounded-full border px-3 py-1 text-xs font-medium ${style.pill}`}>
              {currentLabel}
            </span>
            <div className="text-right text-xs text-slate-400">
              <div>
                <span className="text-slate-500">Elapsed</span>{" "}
                <span className="font-mono text-slate-200">{elapsed(startedAt, agents)}</span>
              </div>
              <div>
                <span className="text-slate-500">Progress</span>{" "}
                <span className="font-mono text-slate-200">
                  {done}/{total}
                </span>
              </div>
            </div>
          </div>
        </div>
        <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full bg-emerald-500/70 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {awaiting && (
        <div className="rounded-lg border border-purple-500/40 bg-purple-500/10 px-4 py-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="text-sm text-purple-100">
              <span className="font-semibold">Approval required:</span>{" "}
              <span className="font-mono">{formatDisplayName(awaiting)}</span>
              {awaiting === "ppg_mapping" && (
                <span className="ml-2 text-xs text-purple-300/80">
                  scroll down to pick the modelling grain, or approve with the default below.
                </span>
              )}
              {awaiting !== "ppg_mapping" && (
                <span className="ml-2 text-xs text-purple-300/80">
                  review the agent output below, then approve or reject.
                </span>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleApprove}
                disabled={busy}
                className="rounded border border-emerald-400/60 bg-emerald-500/25 px-4 py-1.5 text-sm font-semibold text-emerald-100 hover:bg-emerald-500/40 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {busy ? "…" : "Approve ✓"}
              </button>
              <button
                type="button"
                onClick={handleReject}
                disabled={busy}
                className="rounded border border-rose-400/60 bg-rose-500/20 px-4 py-1.5 text-sm font-medium text-rose-200 hover:bg-rose-500/30 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Reject ✗
              </button>
              <button
                type="button"
                onClick={jumpToAwaiting}
                className="rounded border border-purple-400/60 bg-purple-500/20 px-3 py-1.5 text-sm font-medium text-purple-100 hover:bg-purple-500/30"
              >
                {awaiting === "ppg_mapping" ? "Configure grain ↓" : "View details ↓"}
              </button>
            </div>
          </div>
          {actionError && (
            <p className="mt-2 text-xs text-rose-300">{actionError}</p>
          )}
        </div>
      )}
    </div>
  );
}
