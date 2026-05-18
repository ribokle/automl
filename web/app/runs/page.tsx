import Link from "next/link";
import { listRuns } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function RunsPage() {
  let runs: Awaited<ReturnType<typeof listRuns>> = [];
  try {
    runs = await listRuns();
  } catch {
    runs = [];
  }
  return (
    <main className="space-y-6">
      <h1 className="text-2xl font-bold">Runs</h1>
      <div className="rounded-lg border border-slate-800 bg-slate-900/60">
        {runs.length === 0 ? (
          <p className="p-6 text-slate-400">No runs yet. Go to the home page to start one.</p>
        ) : (
          <ul className="divide-y divide-slate-800">
            {runs.map((r) => (
              <li key={r.id} className="flex items-center justify-between p-4">
                <div>
                  <Link href={`/runs/${r.id}`} className="font-mono text-emerald-300 hover:underline">
                    {r.id}
                  </Link>
                  <p className="text-xs text-slate-500">{r.data_path}</p>
                </div>
                <span className="rounded bg-slate-800 px-2 py-0.5 text-xs">{r.status}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </main>
  );
}
