"use client";

import type { AgentName, AgentState } from "@/lib/types";
import { AGENT_ORDER } from "@/lib/types";

interface Props {
  agents: Record<AgentName, AgentState> | null;
}

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${n}`;
}

function fmtUsd(n: number): string {
  if (n === 0) return "$0.0000";
  if (n < 0.001) return `$${n.toExponential(1)}`;
  return `$${n.toFixed(4)}`;
}

function fmtDuration(start?: string | null, end?: string | null): string {
  if (!start || !end) return "—";
  const t0 = Date.parse(start);
  const t1 = Date.parse(end);
  if (!Number.isFinite(t0) || !Number.isFinite(t1)) return "—";
  const ms = Math.max(0, t1 - t0);
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

const STATUS_COLOR: Record<string, string> = {
  done: "text-emerald-300",
  running: "text-amber-300",
  awaiting_approval: "text-purple-300",
  failed: "text-rose-300",
  skipped: "text-slate-500",
  pending: "text-slate-500",
};

export function CostDashboard({ agents }: Props) {
  if (!agents) return null;
  const rows = AGENT_ORDER.map((name) => {
    const a = agents[name];
    return {
      agent: name,
      status: a?.status ?? "pending",
      tokens_in: a?.tokens_in ?? 0,
      tokens_out: a?.tokens_out ?? 0,
      cost_usd: a?.cost_usd ?? 0,
      duration: fmtDuration(a?.started_at, a?.finished_at),
    };
  });

  const totals = rows.reduce(
    (acc, r) => ({
      tokens_in: acc.tokens_in + r.tokens_in,
      tokens_out: acc.tokens_out + r.tokens_out,
      cost_usd: acc.cost_usd + r.cost_usd,
    }),
    { tokens_in: 0, tokens_out: 0, cost_usd: 0 },
  );
  const anyTokens = totals.tokens_in + totals.tokens_out > 0;

  return (
    <details className="rounded-xl border border-slate-800 bg-slate-900/60">
      <summary className="cursor-pointer list-none px-5 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-[10px] uppercase tracking-wider text-slate-500">Cost &amp; runtime</p>
            <h2 className="text-sm font-semibold text-slate-200">
              Per-agent tokens · USD · duration
            </h2>
          </div>
          <div className="flex gap-4 font-mono text-xs">
            <span className="text-slate-400">
              <span className="text-slate-500">tokens</span>{" "}
              <span className="text-slate-200">
                {fmtTokens(totals.tokens_in)} in · {fmtTokens(totals.tokens_out)} out
              </span>
            </span>
            <span className="text-slate-400">
              <span className="text-slate-500">cost</span>{" "}
              <span className="text-slate-200">{fmtUsd(totals.cost_usd)}</span>
            </span>
          </div>
        </div>
      </summary>
      <div className="overflow-x-auto border-t border-slate-800 px-5 py-3">
        {!anyTokens && (
          <p className="mb-2 text-[10.5px] text-slate-500">
            No LLM tokens were recorded — this run was executed in dry-run mode
            (no <code>ANTHROPIC_API_KEY</code>). Durations are still real.
          </p>
        )}
        <table className="w-full text-left text-[11px]">
          <thead>
            <tr className="border-b border-slate-800 text-[10px] uppercase tracking-wider text-slate-500">
              <th className="py-1.5 pr-3">Agent</th>
              <th className="py-1.5 pr-3">Status</th>
              <th className="py-1.5 pr-3 text-right">Tokens in</th>
              <th className="py-1.5 pr-3 text-right">Tokens out</th>
              <th className="py-1.5 pr-3 text-right">Cost</th>
              <th className="py-1.5 pr-3 text-right">Duration</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.agent} className="border-b border-slate-900">
                <td className="py-1 pr-3 text-slate-200">{r.agent}</td>
                <td className={`py-1 pr-3 ${STATUS_COLOR[r.status] ?? "text-slate-400"}`}>
                  {r.status}
                </td>
                <td className="py-1 pr-3 text-right font-mono text-slate-300">
                  {r.tokens_in ? fmtTokens(r.tokens_in) : "—"}
                </td>
                <td className="py-1 pr-3 text-right font-mono text-slate-300">
                  {r.tokens_out ? fmtTokens(r.tokens_out) : "—"}
                </td>
                <td className="py-1 pr-3 text-right font-mono text-slate-300">
                  {r.cost_usd ? fmtUsd(r.cost_usd) : "—"}
                </td>
                <td className="py-1 pr-3 text-right font-mono text-slate-300">{r.duration}</td>
              </tr>
            ))}
            <tr className="border-t border-slate-700">
              <td className="py-1 pr-3 font-semibold text-slate-200">Total</td>
              <td className="py-1 pr-3" />
              <td className="py-1 pr-3 text-right font-mono text-slate-100">
                {fmtTokens(totals.tokens_in)}
              </td>
              <td className="py-1 pr-3 text-right font-mono text-slate-100">
                {fmtTokens(totals.tokens_out)}
              </td>
              <td className="py-1 pr-3 text-right font-mono text-slate-100">
                {fmtUsd(totals.cost_usd)}
              </td>
              <td className="py-1 pr-3" />
            </tr>
          </tbody>
        </table>
      </div>
    </details>
  );
}
