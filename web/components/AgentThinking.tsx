"use client";

import { useEffect, useState } from "react";
import { getArtifact } from "@/lib/api";
import type { AgentName } from "@/lib/types";

interface LLMCall {
  label: string;
  model: string;
  system: string;
  user: string;
  response: string;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
  dry_run: boolean;
  ts?: string;
}

interface TraceBlob {
  agent: string;
  calls: LLMCall[];
}

interface Props {
  runId: string;
  agent: AgentName;
  ready: boolean;
}

export function AgentThinking({ runId, agent, ready }: Props) {
  const [trace, setTrace] = useState<TraceBlob | null>(null);
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    getArtifact<TraceBlob>(runId, `${agent}_llm_trace.json`)
      .then((d) => {
        if (!cancelled) setTrace(d);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [runId, agent, ready]);

  if (!trace || trace.calls.length === 0) return null;

  return (
    <section className="mb-3">
      <div className="mb-1 flex items-center gap-2">
        <span className="text-[10px] uppercase tracking-wider text-slate-500">Agent thinking</span>
        <span className="text-[10px] text-slate-600">
          {trace.calls.length} call{trace.calls.length === 1 ? "" : "s"}
        </span>
      </div>
      <div className="space-y-2">
        {trace.calls.map((c, i) => (
          <Call key={i} call={c} />
        ))}
      </div>
    </section>
  );
}

function Call({ call }: { call: LLMCall }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded border border-slate-800 bg-slate-900/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-2 py-1.5 text-left text-[11px]"
      >
        <span className="flex items-center gap-2">
          <span className="font-mono text-slate-300">{call.label}</span>
          <span className="font-mono text-[10px] text-slate-500">{call.model}</span>
          {call.dry_run ? (
            <span className="rounded border border-amber-500/40 bg-amber-500/15 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-amber-300">
              dry-run
            </span>
          ) : (
            <span className="text-[10px] text-slate-500">
              {call.tokens_in}↓ {call.tokens_out}↑ · ${call.cost_usd.toFixed(4)}
            </span>
          )}
        </span>
        <span className="text-slate-500">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-slate-800 px-2 py-2 text-[11px]">
          <Pane title="system" body={call.system} />
          <Pane title="user" body={call.user} />
          <Pane title="response" body={call.response} tone="response" />
        </div>
      )}
    </div>
  );
}

function Pane({ title, body, tone }: { title: string; body: string; tone?: "response" }) {
  return (
    <div>
      <div className="mb-0.5 flex items-center justify-between">
        <span className="text-[9px] uppercase tracking-wider text-slate-500">{title}</span>
        <span className="text-[9px] text-slate-600">{body.length} chars</span>
      </div>
      <pre
        className={`max-h-44 overflow-auto whitespace-pre-wrap rounded border border-slate-800 bg-slate-950/60 p-2 font-mono text-[10px] leading-snug ${
          tone === "response" ? "text-emerald-200" : "text-slate-300"
        }`}
      >
        {body || "(empty)"}
      </pre>
    </div>
  );
}
