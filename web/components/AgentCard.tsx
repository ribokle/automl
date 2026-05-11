import type { AgentName, AgentStatus, RunEvent } from "@/lib/types";

const STATUS_COLOR: Record<AgentStatus | "idle", string> = {
  idle: "bg-slate-700 text-slate-300",
  pending: "bg-slate-700 text-slate-300",
  running: "bg-amber-500/20 text-amber-300 border-amber-500/40",
  awaiting_approval: "bg-purple-500/20 text-purple-300 border-purple-500/40",
  done: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  failed: "bg-rose-500/20 text-rose-300 border-rose-500/40",
  skipped: "bg-slate-700/60 text-slate-400",
};

interface Props {
  agent: AgentName;
  status: AgentStatus | "idle";
  index: number;
  events: RunEvent[];
}

export function AgentCard({ agent, status, index, events }: Props) {
  const last = events[events.length - 1];
  return (
    <div className={`rounded-lg border border-slate-800 bg-slate-900/60 p-4 shadow-sm`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-slate-800 text-xs text-slate-400">
            {index + 1}
          </div>
          <h3 className="text-sm font-semibold capitalize text-slate-100">
            {agent.replace(/_/g, " ")}
          </h3>
        </div>
        <span className={`rounded border px-2 py-0.5 text-xs ${STATUS_COLOR[status]}`}>{status}</span>
      </div>
      {events.length > 0 && (
        <div className="mt-3 space-y-1 text-xs text-slate-400">
          {events.slice(-3).map((e, i) => (
            <div key={i} className="font-mono">
              {e.type}
              {e.tool ? ` · ${e.tool}` : ""}
              {e.rows ? ` · rows=${e.rows}` : ""}
              {e.checks ? ` · checks=${e.checks}` : ""}
              {e.outputs ? ` · ${JSON.stringify(e.outputs)}` : ""}
            </div>
          ))}
        </div>
      )}
      {last?.error && (
        <div className="mt-2 rounded bg-rose-950/40 px-2 py-1 text-xs text-rose-300">{last.error}</div>
      )}
    </div>
  );
}
