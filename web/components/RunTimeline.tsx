"use client";

import { AgentCard } from "./AgentCard";
import { useRunEvents } from "@/lib/sse";
import { AGENT_ORDER, type AgentName, type AgentStatus } from "@/lib/types";

interface Props {
  runId: string;
}

export function RunTimeline({ runId }: Props) {
  const events = useRunEvents(runId);
  const byAgent = new Map<AgentName, typeof events>();
  for (const a of AGENT_ORDER) byAgent.set(a, []);
  for (const e of events) {
    if (e.agent && byAgent.has(e.agent)) {
      byAgent.get(e.agent)!.push(e);
    }
  }

  function statusOf(agent: AgentName): AgentStatus | "idle" {
    const evts = byAgent.get(agent) ?? [];
    for (let i = evts.length - 1; i >= 0; i--) {
      const e = evts[i];
      if (e.type === "agent_finished") return (e.status as AgentStatus) ?? "done";
      if (e.type === "agent_failed") return "failed";
      if (e.type === "agent_started") return "running";
    }
    return "idle";
  }

  return (
    <div className="flex flex-col gap-3">
      {AGENT_ORDER.map((agent, i) => (
        <AgentCard
          key={agent}
          agent={agent}
          index={i}
          status={statusOf(agent)}
          events={byAgent.get(agent) ?? []}
        />
      ))}
    </div>
  );
}
