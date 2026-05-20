"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { relativeTime } from "@/lib/agent-meta";
import { archiveRun, deleteRun, listRuns, unarchiveRun } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

type View = "active" | "archived";

const STATUS_STYLE: Record<string, string> = {
  completed: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  running: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  awaiting_approval: "border-purple-500/40 bg-purple-500/10 text-purple-300",
  failed: "border-rose-500/40 bg-rose-500/10 text-rose-300",
  pending: "border-slate-700 bg-slate-800 text-slate-400",
};

export default function RunsPage() {
  const [view, setView] = useState<View>("active");
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const rows = await listRuns(view === "archived");
      setRuns(rows);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "could not load runs");
    }
  }, [view]);

  useEffect(() => {
    setRuns(null);
    void refresh();
    const id = window.setInterval(refresh, 4000);
    return () => window.clearInterval(id);
  }, [refresh]);

  async function onArchive(id: string) {
    setBusyId(id);
    try {
      await archiveRun(id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "archive failed");
    } finally {
      setBusyId(null);
    }
  }

  async function onUnarchive(id: string) {
    setBusyId(id);
    try {
      await unarchiveRun(id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "unarchive failed");
    } finally {
      setBusyId(null);
    }
  }

  async function onDelete(id: string) {
    if (!window.confirm(`Permanently delete run ${id} and all its artifacts?`)) return;
    setBusyId(id);
    try {
      await deleteRun(id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "delete failed");
    } finally {
      setBusyId(null);
    }
  }

  const isArchivedView = view === "archived";

  return (
    <main className="space-y-6">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold text-slate-100">Runs</h1>
        <Link
          href="/"
          className="rounded border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-200 hover:bg-emerald-500/20"
        >
          + New run
        </Link>
      </div>

      <div className="flex gap-1 rounded border border-slate-800 bg-slate-950 p-1 text-xs">
        <button
          type="button"
          onClick={() => setView("active")}
          className={`flex-1 rounded px-3 py-1.5 font-medium ${
            view === "active"
              ? "bg-emerald-500/15 text-emerald-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Active
        </button>
        <button
          type="button"
          onClick={() => setView("archived")}
          className={`flex-1 rounded px-3 py-1.5 font-medium ${
            view === "archived"
              ? "bg-emerald-500/15 text-emerald-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Archived
        </button>
      </div>

      {error && (
        <div className="rounded border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-xs text-rose-200">
          {error}
        </div>
      )}

      <div className="rounded-lg border border-slate-800 bg-slate-900/60">
        {runs === null ? (
          <p className="px-4 py-6 text-xs text-slate-500">loading…</p>
        ) : runs.length === 0 ? (
          <div className="flex flex-col items-center gap-2 p-10 text-center">
            <p className="text-sm text-slate-300">
              {isArchivedView ? "Archive is empty." : "No runs yet."}
            </p>
            <p className="text-xs text-slate-500">
              {isArchivedView
                ? "Archived runs you delete will be permanently removed from disk."
                : "Start a run from the home page — upload a CSV or use the bundled synthetic panel."}
            </p>
          </div>
        ) : (
          <ul className="divide-y divide-slate-800">
            {runs.map((r) => {
              const style = STATUS_STYLE[r.status] ?? STATUS_STYLE.pending;
              const isBusy = busyId === r.id;
              const isActiveStatus =
                r.status === "running" || r.status === "awaiting_approval";
              return (
                <li key={r.id} className="flex items-center justify-between gap-3 p-4">
                  <div className="min-w-0 flex-1">
                    <Link
                      href={`/runs/${r.id}`}
                      className="font-mono text-sm text-emerald-300 hover:underline"
                    >
                      {r.id}
                    </Link>
                    <p className="truncate text-[11px] text-slate-500">{r.data_path}</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-3">
                    <span className="font-mono text-[11px] text-slate-500">
                      {relativeTime(r.created_at)} ago
                    </span>
                    <span
                      className={`rounded border px-2 py-0.5 text-[10px] uppercase tracking-wider ${style}`}
                    >
                      {r.status.replace(/_/g, " ")}
                    </span>
                    {isArchivedView ? (
                      <>
                        <button
                          type="button"
                          onClick={() => onUnarchive(r.id)}
                          disabled={isBusy}
                          className="rounded border border-slate-700 px-2 py-0.5 text-[10px] font-medium text-slate-300 hover:bg-slate-800 disabled:opacity-50"
                        >
                          restore
                        </button>
                        <button
                          type="button"
                          onClick={() => onDelete(r.id)}
                          disabled={isBusy}
                          className="rounded border border-rose-500/40 bg-rose-500/10 px-2 py-0.5 text-[10px] font-medium text-rose-200 hover:bg-rose-500/20 disabled:opacity-50"
                        >
                          delete
                        </button>
                      </>
                    ) : (
                      <button
                        type="button"
                        onClick={() => onArchive(r.id)}
                        disabled={isBusy || isActiveStatus}
                        title={isActiveStatus ? "wait for the run to finish" : "archive"}
                        className="rounded border border-slate-700 px-2 py-0.5 text-[10px] font-medium text-slate-300 hover:bg-slate-800 disabled:opacity-40"
                      >
                        archive
                      </button>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </main>
  );
}
