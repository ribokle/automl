"use client";

import { useMemo, useState } from "react";

import { AgentCard } from "./AgentCard";
import { ArtifactGallery } from "./ArtifactGallery";
import { CostDashboard } from "./CostDashboard";
import { ExecutiveBanner } from "./ExecutiveBanner";
import { ReplayBar } from "./ReplayBar";
import { RunHeader } from "./RunHeader";
import { useRunEvents, useRunState } from "@/lib/sse";
import { AGENT_ORDER, type AgentName, type AgentStatus, type RunEvent } from "@/lib/types";

interface Props {
  runId: string;
}

export function RunTimeline({ runId }: Props) {
  const events = useRunEvents(runId);
  const runState = useRunState(runId, events);
  const [scrubTs, setScrubTs] = useState<string | null>(null);
  const isReplay = scrubTs !== null;

  const visibleEvents = useMemo<RunEvent[]>(() => {
    if (!scrubTs) return events;
    const cutoff = Date.parse(scrubTs);
    return events.filter((e) => {
      const t = Date.parse(e.ts);
      return Number.isFinite(t) ? t <= cutoff : true;
    });
  }, [events, scrubTs]);

  const byAgent = new Map<AgentName, RunEvent[]>();
  for (const a of AGENT_ORDER) byAgent.set(a, []);
  for (const e of visibleEvents) {
    if (e.agent && byAgent.has(e.agent)) {
      byAgent.get(e.agent)!.push(e);
    }
  }

  function statusOf(agent: AgentName): AgentStatus | "idle" {
    // During replay we ignore live runState and derive purely from filtered events.
    if (!isReplay) {
      const fromState = runState?.agents?.[agent]?.status;
      if (fromState && fromState !== "pending") return fromState;
    }
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

  const runStarted =
    events.find((e) => e.type === "run_started")?.ts ?? runState?.created_at ?? null;

  const insightsState = runState?.agents?.insights;
  const insightsReady = !isReplay && insightsState?.status === "done";
  const hasPdf =
    Boolean(insightsState?.artifacts?.some((a) => a.name === "report.pdf")) ||
    insightsState?.outputs?.pdf === true;

  return (
    <div className="flex flex-col gap-6">
      <RunHeader
        runId={runId}
        runStatus={runState?.status ?? "running"}
        agents={runState?.agents ?? null}
        startedAt={runStarted}
      />
      <ExecutiveBanner runId={runId} insightsReady={insightsReady} hasPdf={hasPdf} />
      <ReplayBar events={events} scrubTs={scrubTs} onScrub={setScrubTs} />
      <div>
        {AGENT_ORDER.map((agent, i) => (
          <AgentCard
            key={agent}
            runId={runId}
            agent={agent}
            index={i}
            status={statusOf(agent)}
            events={byAgent.get(agent) ?? []}
            agentState={isReplay ? undefined : runState?.agents?.[agent]}
            isLast={i === AGENT_ORDER.length - 1}
          />
        ))}
      </div>
      <CostDashboard agents={isReplay ? null : runState?.agents ?? null} />
      <ArtifactGallery runId={runId} agents={isReplay ? null : runState?.agents ?? null} />
    </div>
  );
}
