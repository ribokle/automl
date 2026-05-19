import Link from "next/link";
import { listRuns } from "@/lib/api";

export const dynamic = "force-dynamic";

const STATUS_STYLE: Record<string, string> = {
  completed: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  running: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  awaiting_approval: "border-purple-500/40 bg-purple-500/10 text-purple-300",
  failed: "border-rose-500/40 bg-rose-500/10 text-rose-300",
  pending: "border-slate-700 bg-slate-800 text-slate-400",
};

function relativeTime(iso: string): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "—";
  const diff = (Date.now() - t) / 1000;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

export default async function RunsPage() {
  let runs: Awaited<ReturnType<typeof listRuns>> = [];
  let error: string | null = null;
  try {
    runs = await listRuns();
  } catch (e) {
    error = e instanceof Error ? e.message : "could not load runs";
  }

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

      {error && (
        <div className="rounded border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-xs text-rose-200">
          {error}
        </div>
      )}

      <div className="rounded-lg border border-slate-800 bg-slate-900/60">
        {runs.length === 0 && !error ? (
          <div className="flex flex-col items-center gap-2 p-10 text-center">
            <p className="text-sm text-slate-300">No runs yet.</p>
            <p className="text-xs text-slate-500">
              Start a run from the home page — upload a CSV or use the bundled synthetic panel.
            </p>
          </div>
        ) : (
          <ul className="divide-y divide-slate-800">
            {runs.map((r) => {
              const style = STATUS_STYLE[r.status] ?? STATUS_STYLE.pending;
              return (
                <li key={r.id} className="flex items-center justify-between gap-3 p-4">
                  <div className="min-w-0">
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
                      {relativeTime(r.created_at)}
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
      </div>
    </main>
  );
}
