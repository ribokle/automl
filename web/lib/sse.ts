"use client";

import { useEffect, useState } from "react";
import { eventsUrl, getRun } from "./api";
import type { RunEvent, RunStateFull } from "./types";

export function useRunEvents(runId: string | null): RunEvent[] {
  const [events, setEvents] = useState<RunEvent[]>([]);

  useEffect(() => {
    if (!runId) return;
    const source = new EventSource(eventsUrl(runId));
    const handle = (e: MessageEvent) => {
      try {
        const parsed = JSON.parse(e.data) as RunEvent;
        setEvents((prev) => [...prev, parsed]);
      } catch {
        // ignore keepalive payloads
      }
    };
    source.addEventListener("message", handle);
    source.addEventListener("history", handle);
    source.onerror = () => source.close();
    return () => source.close();
  }, [runId]);

  return events;
}

export function useRunState(runId: string | null, events: RunEvent[]): RunStateFull | null {
  const [state, setState] = useState<RunStateFull | null>(null);
  const trigger = events[events.length - 1]?.ts ?? "";
  const eventCount = events.length;

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    getRun(runId)
      .then((s) => {
        if (!cancelled) setState(s);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [runId, trigger, eventCount]);

  return state;
}
