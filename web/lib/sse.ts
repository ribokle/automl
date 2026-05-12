"use client";

import { useEffect, useState } from "react";
import type { RunEvent } from "./types";
import { eventsUrl } from "./api";

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
