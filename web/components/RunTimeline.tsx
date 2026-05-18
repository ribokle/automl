"use client";

import { AgentCard } from "./AgentCard";
import { ArtifactGallery } from "./ArtifactGallery";
import { PPGTable } from "./PPGTable";
import { RunHeader } from "./RunHeader";
import { useRunEvents, useRunState } from "@/lib/sse";
import { AGENT_ORDER, type AgentName, type AgentStatus } from "@/lib/types";

interface Props {
  runId: string;
}

export function RunTimeline({ runId }: Props) {
  const events = useRunEvents(runId);
  const runState = useRunState(runId, events);

  const byAgent = new Map<AgentName, typeof events>();
  for (const a of AGENT_ORDER) byAgent.set(a, []);
  for (const e of events) {
    if (e.agent && byAgent.has(e.agent)) {
      byAgent.get(e.agent)!.push(e);
    }
  }

  function statusOf(agent: AgentName): AgentStatus | "idle" {
    const fromState = runState?.agents?.[agent]?.status;
    if (fromState && fromState !== "pending") return fromState;
    const evts = byAgent.get(agent) ?? [];
    for (let i = evts.length - 1; i >= 0; i--) {
      const e = evts[i];
      if (e.type === "approval_required") return "awaiting_approval";
      if (e.type === "approval_resolved") return e.approved ? "done" : "failed";
      if (e.type === "agent_finished") return (e.status as AgentStatus) ?? "done";
      if (e.type === "agent_failed") return "failed";
      if (e.type === "agent_started") return "running";
    }
    return "idle";
  }

  const runStarted = events.find((e) => e.type === "run_started")?.ts ?? runState?.created_at ?? null;

  return (
    <div className="flex flex-col gap-6">
      <RunHeader
        runId={runId}
        runStatus={runState?.status ?? "running"}
        agents={runState?.agents ?? null}
        startedAt={runStarted}
      />
      <div>
        {AGENT_ORDER.map((agent, i) => (
          <AgentCard
            key={agent}
            runId={runId}
            agent={agent}
            index={i}
            status={statusOf(agent)}
            events={byAgent.get(agent) ?? []}
            agentState={runState?.agents?.[agent]}
            isLast={i === AGENT_ORDER.length - 1}
          />
        ))}
      </div>
      <PPGTable runId={runId} events={events} />
      <ArtifactGallery runId={runId} agents={runState?.agents ?? null} />
    </div>
  );
}
