import { AGENT_ORDER, type AgentName, type AgentState } from "@/lib/types";
import { formatDisplayName, formatDuration, STATUS_STYLE } from "@/lib/agent-meta";

interface Props {
  runId: string;
  runStatus: string;
  agents: Record<AgentName, AgentState> | null;
  startedAt: string | null;
}

function elapsed(start: string | null, agents: Record<AgentName, AgentState> | null): string {
  if (!agents) return formatDuration(start, null) ?? "—";
  const finishedAll = AGENT_ORDER.every((a) => agents[a]?.status === "done" || agents[a]?.status === "skipped");
  const lastEnd = AGENT_ORDER.map((a) => agents[a]?.finished_at).filter(Boolean).sort().pop();
  return formatDuration(start, finishedAll ? (lastEnd as string) : null) ?? "—";
}

export function RunHeader({ runId, runStatus, agents, startedAt }: Props) {
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
    // Scroll to the approval block specifically (not the card top) so the
    // operator lands directly on the buttons. Falls back to the card if
    // the approval block hasn't mounted yet.
    const target =
      document.getElementById(`approval-${awaiting}`) ??
      document.getElementById(`agent-${awaiting}`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
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
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-purple-500/40 bg-purple-500/10 px-4 py-3">
          <div className="text-sm text-purple-100">
            <span className="font-semibold">Approval required:</span>{" "}
            <span className="font-mono">{formatDisplayName(awaiting)}</span>
            <span className="ml-2 text-xs text-purple-300/80">
              the pipeline is paused — review the panel below to continue.
            </span>
          </div>
          <button
            type="button"
            onClick={jumpToAwaiting}
            className="rounded border border-purple-400/60 bg-purple-500/20 px-3 py-1 text-xs font-medium text-purple-100 hover:bg-purple-500/30"
          >
            Jump to approval →
          </button>
        </div>
      )}
    </div>
  );
}
