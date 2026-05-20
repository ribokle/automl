"use client";

import { useEffect, useState } from "react";

import { getHealth } from "@/lib/api";

type Health = "checking" | "up" | "down";

const POLL_MS = 15_000;

export function BackendStatus() {
  const [health, setHealth] = useState<Health>("checking");
  const [lastChecked, setLastChecked] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const ctrl = new AbortController();

    async function probe() {
      try {
        await getHealth(ctrl.signal);
        if (!cancelled) {
          setHealth("up");
          setLastChecked(Date.now());
        }
      } catch {
        if (!cancelled) {
          setHealth("down");
          setLastChecked(Date.now());
        }
      }
    }

    void probe();
    const id = window.setInterval(probe, POLL_MS);
    return () => {
      cancelled = true;
      ctrl.abort();
      window.clearInterval(id);
    };
  }, []);

  const style =
    health === "up"
      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
      : health === "down"
        ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
        : "border-slate-700 bg-slate-800 text-slate-400";

  const label =
    health === "up" ? "API online" : health === "down" ? "API offline" : "checking…";

  const title =
    lastChecked != null
      ? `Last checked ${new Date(lastChecked).toLocaleTimeString()}`
      : "Polling /health";

  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wider ${style}`}
    >
      <span
        className={`inline-block h-1.5 w-1.5 rounded-full ${
          health === "up"
            ? "bg-emerald-400"
            : health === "down"
              ? "bg-rose-400"
              : "bg-slate-500 animate-pulse"
        }`}
      />
      {label}
    </span>
  );
}
