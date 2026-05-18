import {
  BoxPlot,
  CorrHeatmap,
  CoverageGrid,
  PPGScatter,
  TrendChart,
  VIFBar,
} from "../_components/Charts";
import { AnomalyTable, DataPreview, DropLog, QualityPanel, SchemaTable } from "../_components/Tables";

function Pill({ tone, children }: { tone: "done" | "running"; children: React.ReactNode }) {
  const c =
    tone === "done"
      ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/40"
      : "bg-amber-500/15 text-amber-300 border-amber-500/40";
  return (
    <span className={`rounded border px-2 py-0.5 text-[10px] uppercase tracking-wide ${c}`}>{children}</span>
  );
}

function Step({
  n,
  title,
  description,
  pill,
  summary,
  isLast,
  expanded,
  children,
}: {
  n: number;
  title: string;
  description: string;
  pill: "done" | "running";
  summary: string[];
  isLast?: boolean;
  expanded?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <div className="relative pl-10">
      <div className="absolute left-0 top-0 flex h-full flex-col items-center">
        <div
          className={`flex h-7 w-7 items-center justify-center rounded-full ring-4 ${pill === "done" ? "ring-emerald-500/30 bg-emerald-400" : "ring-amber-500/40 bg-amber-400 animate-pulse"} text-[10px] font-bold text-slate-900`}
        >
          {pill === "done" ? "✓" : n}
        </div>
        {!isLast && <div className="mt-1 w-px flex-1 bg-slate-800" />}
      </div>
      <div className="mb-3 rounded-lg border border-slate-800 bg-slate-900/60">
        <div className="flex items-start justify-between gap-3 p-4">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
              <Pill tone={pill}>{pill}</Pill>
            </div>
            <p className="mt-1 text-xs text-slate-400">{description}</p>
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
              {summary.map((s, i) => (
                <span key={i}>{s}</span>
              ))}
            </div>
          </div>
          {expanded && <span className="text-slate-500">▾</span>}
        </div>
        {expanded && children && (
          <div className="border-t border-slate-800 p-4">{children}</div>
        )}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4 last:mb-0">
      <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">{title}</h4>
      {children}
    </div>
  );
}

export default function InlineMockup() {
  return (
    <main>
      <div className="mb-4 rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-xs text-amber-200">
        <strong>Mockup · INLINE layout.</strong> Each AgentCard expands to reveal its data /
        quality / chart artefacts right where the step lives. Three Phase-2a-relevant cards are shown
        expanded; the rest collapse to their existing summary row.
      </div>
      <div className="mb-6 rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs uppercase tracking-wider text-slate-500">Run</p>
            <h1 className="font-mono text-lg">demo-run-7181d48</h1>
          </div>
          <div className="text-right text-xs text-slate-400">
            <span className="rounded-full border border-emerald-500/40 bg-emerald-500/15 px-3 py-1 text-emerald-300">
              completed
            </span>
            <div className="mt-2">
              Progress <span className="font-mono text-slate-200">14/14</span> · Elapsed{" "}
              <span className="font-mono text-slate-200">12.4 s</span>
            </div>
          </div>
        </div>
        <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-slate-800">
          <div className="h-full bg-emerald-500/70" style={{ width: "100%" }} />
        </div>
      </div>

      <Step
        n={1}
        title="Ingestion"
        description="Load CSV into DuckDB, build the dbt panel mart, run schema + distribution checks."
        pill="done"
        expanded
        summary={["24,960 rows loaded", "25 dbt checks", "8 GE checks", "5 anomalies flagged"]}
      >
        <div className="grid gap-5 md:grid-cols-2">
          <Section title="Data preview · main.panel">
            <DataPreview />
          </Section>
          <Section title="Schema · column roles">
            <SchemaTable />
          </Section>
          <Section title="Coverage grid · SKU × week (red = missing)">
            <CoverageGrid />
          </Section>
          <Section title="Quality checks">
            <QualityPanel />
          </Section>
          <Section title="Anomalies (5)">
            <AnomalyTable />
          </Section>
        </div>
      </Step>

      <Step
        n={2}
        title="PPG Mapping"
        description="Group SKUs into Price-Pack Groups; score each by within-group price coherence."
        pill="done"
        expanded
        summary={["8 PPGs", "48 SKUs grouped", "0 flagged", "mean conf 0.92"]}
      >
        <div className="grid gap-5 md:grid-cols-2">
          <Section title="SKU scatter · price tier × log price, coloured by PPG">
            <PPGScatter />
          </Section>
          <Section title="Within-PPG price distribution (box)">
            <BoxPlot />
          </Section>
        </div>
      </Step>

      <Step
        n={5}
        title="EDA"
        description="Panel overview, target-vs-feature relationships, pairwise correlation among numeric features."
        pill="done"
        expanded
        summary={["9 features summarised", "top: price ρ=-0.94", "no missingness"]}
      >
        <div className="grid gap-5 md:grid-cols-2">
          <Section title="Weekly units & average price">
            <TrendChart />
          </Section>
          <Section title="Pairwise correlation (numeric features)">
            <CorrHeatmap />
          </Section>
        </div>
      </Step>

      <Step
        n={7}
        title="Feature Refine"
        description="Iterative VIF + |corr|>0.95 pruning with log_price protected."
        pill="done"
        expanded
        isLast
        summary={["11 kept", "4 dropped", "max VIF 7.92", "max |corr| 0.91", "passes"]}
      >
        <div className="grid gap-5 md:grid-cols-2">
          <Section title="VIF per kept feature">
            <VIFBar />
          </Section>
          <Section title="Drop log">
            <DropLog />
          </Section>
        </div>
      </Step>
    </main>
  );
}
