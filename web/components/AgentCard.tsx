"use client";

import { useState } from "react";
import {
  AGENT_META,
  STATUS_STYLE,
  formatDuration,
  summariseOutputs,
  summariseTool,
} from "@/lib/agent-meta";
import { approveAgent, artifactUrl, rejectAgent } from "@/lib/api";
import type { AgentName, AgentState, AgentStatus, RunEvent } from "@/lib/types";

interface Props {
  runId: string;
  agent: AgentName;
  index: number;
  status: AgentStatus | "idle";
  events: RunEvent[];
  agentState: AgentState | undefined;
  isLast: boolean;
}

export function AgentCard({ runId, agent, index, status, events, agentState, isLast }: Props) {
  const [open, setOpen] = useState(status === "running" || status === "awaiting_approval" || status === "failed");
  const meta = AGENT_META[agent];
  const style = STATUS_STYLE[status];

  const toolCalls = events.filter((e) => e.type === "tool_called");
  const outputs = agentState?.outputs ?? events.find((e) => e.type === "agent_finished")?.outputs ?? null;
  const summary = summariseOutputs(agent, outputs);
  const duration = formatDuration(agentState?.started_at, agentState?.finished_at);
  const reasoning = agentState?.reasoning;
  const confidence = agentState?.confidence;
  const errorText = agentState?.error ?? events.find((e) => e.type === "agent_failed")?.error;
  const showDisclosure = Boolean(reasoning || toolCalls.length > 0 || errorText || (agentState?.artifacts?.length ?? 0) > 0);

  async function handleApprove() {
    await approveAgent(runId, agent);
  }
  async function handleReject() {
    await rejectAgent(runId, agent);
  }

  return (
    <div className="relative pl-10">
      <div className="absolute left-0 top-0 flex h-full flex-col items-center">
        <div
          className={`flex h-7 w-7 items-center justify-center rounded-full ring-4 ${style.ring} ${style.dot} text-[10px] font-bold text-slate-900`}
        >
          {status === "done" ? "✓" : status === "failed" ? "!" : index + 1}
        </div>
        {!isLast && <div className="mt-1 w-px flex-1 bg-slate-800" />}
      </div>

      <div className={`mb-3 rounded-lg border border-slate-800 bg-slate-900/60 transition`}>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex w-full items-start justify-between gap-3 p-4 text-left"
        >
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-semibold text-slate-100">{meta.title}</h3>
              <span className={`rounded border px-2 py-0.5 text-[10px] uppercase tracking-wide ${style.pill}`}>
                {status === "idle" ? "pending" : status}
              </span>
              {typeof confidence === "number" && status === "done" && (
                <span className="rounded border border-slate-700 bg-slate-800/70 px-2 py-0.5 text-[10px] text-slate-300">
                  conf {confidence.toFixed(2)}
                </span>
              )}
              {duration && (
                <span className="text-[10px] text-slate-500">{duration}</span>
              )}
            </div>
            <p className="mt-1 text-xs text-slate-400">{meta.description}</p>
            {summary.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-300">
                {summary.map((s, i) => (
                  <span key={i} className="font-mono">
                    {s}
                  </span>
                ))}
              </div>
            )}
          </div>
          {showDisclosure && (
            <span className="mt-1 text-xs text-slate-500">{open ? "▾" : "▸"}</span>
          )}
        </button>

        {open && showDisclosure && (
          <div className="border-t border-slate-800 px-4 py-3 text-xs">
            {reasoning && (
              <div className="mb-3">
                <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">Reasoning</div>
                <p className="text-slate-300">{reasoning}</p>
              </div>
            )}
            {toolCalls.length > 0 && (
              <div className="mb-3">
                <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">Tool calls</div>
                <ul className="space-y-1 font-mono text-[11px] text-slate-300">
                  {toolCalls.map((e, i) => (
                    <li key={i} className="flex justify-between gap-3">
                      <span>{summariseTool(e.tool ?? "?", e as unknown as Record<string, unknown>)}</span>
                      <span className="text-slate-500">{new Date(e.ts).toLocaleTimeString()}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {agentState?.artifacts && agentState.artifacts.length > 0 && (
              <div className="mb-3">
                <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">Artifacts</div>
                <ul className="space-y-0.5 font-mono text-[11px]">
                  {agentState.artifacts.map((a) => (
                    <li key={a.id}>
                      <a
                        href={artifactUrl(runId, a.name)}
                        target="_blank"
                        rel="noreferrer"
                        className="text-emerald-300 hover:underline"
                      >
                        {a.name}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {errorText && (
              <div className="rounded bg-rose-950/40 px-2 py-1.5 text-[11px] text-rose-300">{errorText}</div>
            )}
          </div>
        )}

        {status === "awaiting_approval" && (
          <div className="flex items-center justify-between gap-2 border-t border-purple-500/30 bg-purple-500/5 px-4 py-2">
            <span className="text-xs text-purple-200">Approval required to proceed</span>
            <div className="flex gap-2">
              <button
                onClick={handleApprove}
                className="rounded bg-emerald-500/20 px-3 py-1 text-xs font-medium text-emerald-200 hover:bg-emerald-500/30"
              >
                Approve
              </button>
              <button
                onClick={handleReject}
                className="rounded bg-rose-500/20 px-3 py-1 text-xs font-medium text-rose-200 hover:bg-rose-500/30"
              >
                Reject
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
