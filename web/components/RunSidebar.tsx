"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { relativeTime } from "@/lib/agent-meta";
import { listRuns } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

const STATUS_DOT: Record<string, string> = {
  completed: "bg-emerald-400",
  running: "bg-amber-400",
  awaiting_approval: "bg-purple-400",
  failed: "bg-rose-500",
  pending: "bg-slate-500",
};

interface Props {
  activeRunId: string;
}

export function RunSidebar({ activeRunId }: Props) {
  const [open, setOpen] = useState(true);
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const list = await listRuns();
        if (!cancelled) setRuns(list);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "failed");
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (!open) {
    return (
      <aside className="sticky top-4 self-start">
        <button
          type="button"
          onClick={() => setOpen(true)}
          aria-label="open run sidebar"
          className="rounded border border-slate-800 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-300 hover:bg-slate-800"
        >
          ▸ runs
        </button>
      </aside>
    );
  }

  const filtered = runs ?? [];

  return (
    <aside className="sticky top-4 w-60 shrink-0 self-start">
      <div className="rounded-lg border border-slate-800 bg-slate-900/60">
        <header className="flex items-center justify-between border-b border-slate-800 px-3 py-2">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Runs
            </span>
            <span className="rounded border border-slate-700 px-1 font-mono text-[10px] text-slate-400">
              {filtered.length}
            </span>
          </div>
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="collapse"
            className="text-[11px] text-slate-500 hover:text-slate-300"
          >
            ◂
          </button>
        </header>
        <div className="max-h-[70vh] overflow-y-auto">
          {error && (
            <p className="px-3 py-2 text-[11px] text-rose-300">{error}</p>
          )}
          {!error && runs === null && (
            <p className="px-3 py-2 text-[11px] text-slate-500">loading…</p>
          )}
          {runs !== null && filtered.length === 0 && (
            <p className="px-3 py-2 text-[11px] text-slate-500">No other runs.</p>
          )}
          <ul className="divide-y divide-slate-800/70">
            {filtered.map((r) => {
              const active = r.id === activeRunId;
              return (
                <li key={r.id}>
                  <Link
                    href={`/runs/${r.id}`}
                    className={`flex items-center gap-2 px-3 py-2 text-[11px] hover:bg-slate-800/50 ${
                      active ? "bg-emerald-500/5" : ""
                    }`}
                  >
                    <span
                      className={`mt-0.5 h-2 w-2 shrink-0 rounded-full ${
                        STATUS_DOT[r.status] ?? STATUS_DOT.pending
                      }`}
                      title={r.status}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline justify-between gap-2">
                        <span
                          className={`truncate font-mono ${
                            active ? "text-emerald-200" : "text-slate-200"
                          }`}
                        >
                          {r.id}
                        </span>
                        <span className="font-mono text-[10px] text-slate-500">
                          {relativeTime(r.created_at)}
                        </span>
                      </div>
                      <span className="truncate text-[10px] text-slate-500">
                        {r.data_path}
                      </span>
                    </div>
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
        <footer className="border-t border-slate-800 px-3 py-2 text-right">
          <Link
            href="/"
            className="text-[10px] font-medium text-emerald-300 hover:underline"
          >
            + new run
          </Link>
        </footer>
      </div>
    </aside>
  );
}
