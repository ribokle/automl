"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createRun } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [dataPath, setDataPath] = useState("data/synthetic.csv");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function start() {
    setBusy(true);
    setError(null);
    try {
      const run = await createRun(dataPath, false);
      router.push(`/runs/${run.id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="space-y-8">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">AutoML — Agentic Pricing</h1>
        <p className="text-slate-400">End-to-end price &amp; promo optimization with reasoning per stage.</p>
      </header>

      <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-6">
        <h2 className="mb-4 text-lg font-semibold">Start a new run</h2>
        <label className="block text-sm text-slate-400">CSV path (server-side)</label>
        <input
          className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-sm"
          value={dataPath}
          onChange={(e) => setDataPath(e.target.value)}
          placeholder="data/synthetic.csv"
        />
        <button
          onClick={start}
          disabled={busy}
          className="mt-4 rounded bg-emerald-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-emerald-400 disabled:opacity-50"
        >
          {busy ? "Starting..." : "Run pipeline"}
        </button>
        {error && <p className="mt-3 text-sm text-rose-400">{error}</p>}
      </section>

      <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-6">
        <h2 className="text-lg font-semibold">Quickstart</h2>
        <ol className="mt-3 list-decimal space-y-1 pl-6 text-sm text-slate-300">
          <li><code>make seed</code> — generate the synthetic dataset.</li>
          <li><code>make dbt-deps</code> — install dbt packages.</li>
          <li><code>make api</code> &amp; <code>make web</code> — launch backend + frontend.</li>
          <li>Click <em>Run pipeline</em>.</li>
        </ol>
      </section>
    </main>
  );
}
