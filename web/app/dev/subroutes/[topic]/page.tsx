import Link from "next/link";
import {
  BoxPlot,
  CorrHeatmap,
  CoverageGrid,
  PPGScatter,
  TrendChart,
  VIFBar,
} from "../../_components/Charts";
import {
  AnomalyTable,
  DataPreview,
  DropLog,
  QualityPanel,
  SchemaTable,
} from "../../_components/Tables";

const NAV = [
  { slug: "pipeline", label: "Pipeline" },
  { slug: "data", label: "Data" },
  { slug: "quality", label: "Quality" },
  { slug: "ppg", label: "PPG" },
  { slug: "features", label: "Features" },
];

function Section({ title, children, span }: { title: string; children: React.ReactNode; span?: 1 | 2 }) {
  return (
    <div className={`rounded-lg border border-slate-800 bg-slate-900/40 p-3 ${span === 2 ? "md:col-span-2" : ""}`}>
      <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">{title}</h4>
      {children}
    </div>
  );
}

export function generateStaticParams() {
  return NAV.map((n) => ({ topic: n.slug }));
}

export default function SubroutePage({ params }: { params: { topic: string } }) {
  const t = params.topic;
  return (
    <main>
      <div className="mb-4 rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-xs text-amber-200">
        <strong>Mockup · SUB-ROUTE layout.</strong> Each topic is its own page —{" "}
        <code>/runs/[id]/data</code>, <code>/quality</code>, <code>/ppg</code>, <code>/features</code>.
        Pipeline page stays minimal. Currently showing <code>/{t}</code>.
      </div>
      <div className="mb-6 rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs uppercase tracking-wider text-slate-500">Run</p>
            <h1 className="font-mono text-lg">demo-run-7181d48</h1>
          </div>
          <span className="rounded-full border border-emerald-500/40 bg-emerald-500/15 px-3 py-1 text-xs text-emerald-300">
            completed · 14/14
          </span>
        </div>
        <nav className="mt-5 flex flex-wrap gap-2 border-t border-slate-800 pt-4">
          {NAV.map((n) => (
            <Link
              key={n.slug}
              href={`/dev/subroutes/${n.slug}`}
              className={`rounded border px-3 py-1 text-xs ${
                n.slug === t
                  ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
                  : "border-slate-700 text-slate-400 hover:text-slate-200"
              }`}
            >
              {n.label}
            </Link>
          ))}
        </nav>
      </div>

      {t === "data" && (
        <div className="grid gap-4 md:grid-cols-2">
          <Section title="main.panel preview" span={2}>
            <DataPreview />
          </Section>
          <Section title="Schema">
            <SchemaTable />
          </Section>
          <Section title="Coverage SKU × week">
            <CoverageGrid />
          </Section>
          <Section title="Weekly trend" span={2}>
            <TrendChart />
          </Section>
        </div>
      )}
      {t === "quality" && (
        <div className="grid gap-4 md:grid-cols-2">
          <Section title="dbt + GE checks">
            <QualityPanel />
          </Section>
          <Section title="Anomalies flagged by ingestion">
            <AnomalyTable />
          </Section>
        </div>
      )}
      {t === "ppg" && (
        <div className="grid gap-4 md:grid-cols-2">
          <Section title="SKU scatter — price tier × log price">
            <PPGScatter />
          </Section>
          <Section title="Within-PPG price distribution">
            <BoxPlot />
          </Section>
        </div>
      )}
      {t === "features" && (
        <div className="grid gap-4 md:grid-cols-2">
          <Section title="VIF per kept feature">
            <VIFBar />
          </Section>
          <Section title="Pairwise correlation">
            <CorrHeatmap />
          </Section>
          <Section title="Drop log">
            <DropLog />
          </Section>
        </div>
      )}
      {t === "pipeline" && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3 text-xs text-slate-300">
          (Pipeline view stays minimal — only the 14-step list. Deep dives are in the sister pages.)
        </div>
      )}
    </main>
  );
}
