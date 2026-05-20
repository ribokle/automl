"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { createRun, listRuns, uploadCsv } from "@/lib/api";
import { relativeTime } from "@/lib/agent-meta";
import type { RunSummary } from "@/lib/types";

type Mode = "upload" | "synthetic";

const STATUS_STYLE: Record<string, string> = {
  completed: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  running: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  awaiting_approval: "border-purple-500/40 bg-purple-500/10 text-purple-300",
  failed: "border-rose-500/40 bg-rose-500/10 text-rose-300",
  pending: "border-slate-700 bg-slate-800 text-slate-400",
};

export default function Home() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [mode, setMode] = useState<Mode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [syntheticPath, setSyntheticPath] = useState("data/synthetic.csv");
  const [gatesEnabled, setGatesEnabled] = useState(false);
  const [agentMode, setAgentMode] = useState(true);

  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [recent, setRecent] = useState<RunSummary[]>([]);
  const [recentError, setRecentError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const rows = await listRuns();
        if (!cancelled) setRecent(rows.slice(0, 5));
      } catch (e) {
        if (!cancelled) setRecentError(e instanceof Error ? e.message : String(e));
      }
    }
    void load();
    const id = window.setInterval(load, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  async function start() {
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      let dataPath: string;
      if (mode === "upload") {
        if (!file) {
          setError("Choose a CSV first.");
          setBusy(false);
          return;
        }
        setStatus(`Uploading ${file.name}…`);
        const result = await uploadCsv(file);
        dataPath = result.path;
      } else {
        if (!syntheticPath.trim()) {
          setError("Enter a server-side path first.");
          setBusy(false);
          return;
        }
        dataPath = syntheticPath;
      }
      setStatus("Starting run…");
      const run = await createRun(dataPath, gatesEnabled, agentMode);
      router.push(`/runs/${run.id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
      setStatus(null);
    }
  }

  const selectedLabel =
    mode === "upload"
      ? file
        ? `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MiB)`
        : null
      : syntheticPath.trim() || null;

  return (
    <main className="space-y-8">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight">Start a new run</h1>
        <p className="text-sm text-slate-400">
          Upload a weekly panel CSV or use the bundled synthetic dataset. The 14-agent
          pipeline ingests, builds PPGs, fits elasticity, simulates, and optimises.
        </p>
      </header>

      <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-6">
        <div className="mb-4 flex gap-1 rounded border border-slate-800 bg-slate-950 p-1 text-xs">
          <button
            type="button"
            onClick={() => setMode("upload")}
            className={`flex-1 rounded px-3 py-1.5 font-medium ${
              mode === "upload"
                ? "bg-emerald-500/15 text-emerald-200"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Upload CSV
          </button>
          <button
            type="button"
            onClick={() => setMode("synthetic")}
            className={`flex-1 rounded px-3 py-1.5 font-medium ${
              mode === "synthetic"
                ? "bg-emerald-500/15 text-emerald-200"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Server-side path
          </button>
        </div>

        <div className={mode === "upload" ? "" : "hidden"}>
          <label className="block text-xs text-slate-400">
            CSV file (required columns: <code>sku, store_id, week_start, units, price, tpr_flag</code>)
          </label>
          <div className="mt-2 flex items-center gap-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv,application/csv,application/vnd.ms-excel"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm text-slate-300 file:mr-3 file:rounded file:border-0 file:bg-slate-800 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-slate-200 hover:file:bg-slate-700"
            />
            {file && (
              <button
                type="button"
                onClick={() => {
                  setFile(null);
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
                className="shrink-0 rounded border border-slate-700 px-2 py-1 text-[11px] text-slate-400 hover:text-slate-200"
              >
                clear
              </button>
            )}
          </div>
        </div>

        <div className={mode === "synthetic" ? "" : "hidden"}>
          <label className="block text-xs text-slate-400">
            CSV path on the API server filesystem
          </label>
          <input
            className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-sm"
            value={syntheticPath}
            onChange={(e) => setSyntheticPath(e.target.value)}
            placeholder="data/synthetic.csv"
          />
          <p className="mt-2 text-[11px] text-slate-500">
            Default points at the bundled panel. Run <code>make seed</code> first if absent.
          </p>
        </div>

        <div className="mt-4 rounded border border-slate-800 bg-slate-950/60 px-3 py-2 text-[11px]">
          <span className="text-slate-500">Selected: </span>
          {selectedLabel ? (
            <span className="font-mono text-slate-300">{selectedLabel}</span>
          ) : (
            <span className="italic text-slate-600">none yet</span>
          )}
        </div>

        <div className="mt-4 space-y-2">
          <label className="flex items-start gap-2 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={agentMode}
              onChange={(e) => setAgentMode(e.target.checked)}
              className="mt-0.5 h-3.5 w-3.5 accent-emerald-500"
            />
            <span>
              <span className="font-medium text-slate-200">Agent mode</span> — call the LLM
              for narratives and analyst-style summaries.
              <span className="block text-[10px] text-slate-500">
                Off: every agent uses deterministic fallbacks only (no LLM spend, identical output across runs).
              </span>
            </span>
          </label>
          <label className="flex items-start gap-2 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={gatesEnabled}
              onChange={(e) => setGatesEnabled(e.target.checked)}
              className="mt-0.5 h-3.5 w-3.5 accent-emerald-500"
            />
            <span>
              <span className="font-medium text-slate-200">Approval gates</span> — pause
              after PPG mapping, modeling, and optimization for manual review.
            </span>
          </label>
        </div>

        <div className="mt-5 flex items-center gap-3">
          <button
            onClick={start}
            disabled={busy}
            className="rounded bg-emerald-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-emerald-400 disabled:opacity-50"
          >
            {busy ? "Working…" : "Run pipeline"}
          </button>
          {status && <span className="text-xs text-slate-400">{status}</span>}
        </div>
        {error && <p className="mt-3 text-sm text-rose-400">{error}</p>}
      </section>

      <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-6">
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="text-lg font-semibold">Recent runs</h2>
          <Link href="/runs" className="text-xs text-emerald-300 hover:underline">
            View all →
          </Link>
        </div>
        {recentError ? (
          <p className="text-xs text-rose-300">{recentError}</p>
        ) : recent.length === 0 ? (
          <p className="text-xs text-slate-500">
            No runs yet — kick one off above and it will appear here.
          </p>
        ) : (
          <ul className="divide-y divide-slate-800">
            {recent.map((r) => {
              const style = STATUS_STYLE[r.status] ?? STATUS_STYLE.pending;
              return (
                <li key={r.id} className="flex items-center justify-between gap-3 py-2.5">
                  <div className="min-w-0">
                    <Link
                      href={`/runs/${r.id}`}
                      className="font-mono text-xs text-emerald-300 hover:underline"
                    >
                      {r.id}
                    </Link>
                    <p className="truncate text-[11px] text-slate-500">{r.data_path}</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-3">
                    <span className="font-mono text-[10px] text-slate-500">
                      {relativeTime(r.created_at)} ago
                    </span>
                    <span
                      className={`rounded border px-2 py-0.5 text-[10px] uppercase tracking-wider ${style}`}
                    >
                      {r.status.replace(/_/g, " ")}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-6">
        <h2 className="text-lg font-semibold">Quickstart</h2>
        <ol className="mt-3 list-decimal space-y-1 pl-6 text-sm text-slate-300">
          <li><code>make seed</code> — generate the bundled synthetic panel.</li>
          <li><code>make dbt-deps</code> — install dbt packages.</li>
          <li><code>make api</code> &amp; <code>make web</code> — launch backend + frontend.</li>
          <li>Upload your CSV (or pick the synthetic shortcut) and hit <em>Run pipeline</em>.</li>
        </ol>
      </section>
    </main>
  );
}
