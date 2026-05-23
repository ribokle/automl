import Link from "next/link";
import { AdvancedEDADashboard } from "@/components/AdvancedEDADashboard";

interface PageProps {
  params: { id: string };
}

export default function AdvancedEDAPage({ params }: PageProps) {
  return (
    <main className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <h1 className="text-lg font-semibold text-slate-200">Advanced EDA</h1>
          <p className="text-[11px] text-slate-500">
            Time-series decomposition, structural anomalies, change points, pre-model promo lift, cross-PPG correlation, and Pareto / ABC.
          </p>
        </div>
        <Link
          href={`/runs/${params.id}`}
          className="rounded border border-slate-700 px-3 py-1 text-[11px] text-slate-300 hover:bg-slate-900"
        >
          ← Back to run timeline
        </Link>
      </div>
      <AdvancedEDADashboard runId={params.id} />
    </main>
  );
}
