import { artifactUrl } from "@/lib/api";
import { AGENT_META } from "@/lib/agent-meta";
import type { AgentName, AgentState } from "@/lib/types";

interface Props {
  runId: string;
  agents: Record<AgentName, AgentState> | null;
}

function kindOf(name: string): { label: string; tone: string } {
  const ext = name.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "json":
      return { label: "JSON", tone: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30" };
    case "parquet":
      return { label: "PARQUET", tone: "bg-violet-500/15 text-violet-300 border-violet-500/30" };
    case "csv":
      return { label: "CSV", tone: "bg-sky-500/15 text-sky-300 border-sky-500/30" };
    case "html":
      return { label: "HTML", tone: "bg-amber-500/15 text-amber-300 border-amber-500/30" };
    case "pdf":
      return { label: "PDF", tone: "bg-rose-500/15 text-rose-300 border-rose-500/30" };
    default:
      return { label: ext?.toUpperCase() ?? "FILE", tone: "bg-slate-700/50 text-slate-300 border-slate-700" };
  }
}

export function ArtifactGallery({ runId, agents }: Props) {
  if (!agents) return null;
  const groups: { agent: AgentName; artifacts: AgentState["artifacts"] }[] = [];
  for (const [name, state] of Object.entries(agents) as [AgentName, AgentState][]) {
    if (state.artifacts && state.artifacts.length > 0) {
      groups.push({ agent: name, artifacts: state.artifacts });
    }
  }
  if (groups.length === 0) return null;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold text-slate-200">Artifacts</h2>
        <span className="text-xs text-slate-500">
          {groups.reduce((acc, g) => acc + (g.artifacts?.length ?? 0), 0)} files
        </span>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {groups.map(({ agent, artifacts }) => (
          <div key={agent} className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
            <div className="mb-2 text-xs font-semibold text-slate-300">{AGENT_META[agent]?.title ?? agent}</div>
            <ul className="space-y-1.5">
              {artifacts!.map((a) => {
                const kind = kindOf(a.name);
                return (
                  <li key={a.id} className="flex items-center justify-between gap-2">
                    <a
                      href={artifactUrl(runId, a.name)}
                      target="_blank"
                      rel="noreferrer"
                      className="truncate font-mono text-[11px] text-emerald-300 hover:underline"
                      title={a.name}
                    >
                      {a.name}
                    </a>
                    <span className={`shrink-0 rounded border px-1.5 py-0.5 text-[9px] tracking-wide ${kind.tone}`}>
                      {kind.label}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
