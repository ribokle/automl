"use client";

import { useState } from "react";
import {
  BoxPlot,
  CorrHeatmap,
  CoverageGrid,
  PPGScatter,
  TrendChart,
  VIFBar,
} from "../_components/Charts";
import { AnomalyTable, DataPreview, DropLog, QualityPanel, SchemaTable } from "../_components/Tables";

const TABS = ["Pipeline", "Data", "Quality", "PPG", "Features"] as const;
type Tab = (typeof TABS)[number];

function Section({ title, children, span }: { title: string; children: React.ReactNode; span?: 1 | 2 }) {
  return (
    <div className={`rounded-lg border border-slate-800 bg-slate-900/40 p-3 ${span === 2 ? "md:col-span-2" : ""}`}>
      <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">{title}</h4>
      {children}
    </div>
  );
}

function Pipeline() {
  return (
    <div className="space-y-2 text-xs text-slate-300">
      {[
        ["ingestion", "done", "24,960 rows · 25 dbt checks · 5 anomalies"],
        ["ppg_mapping", "done", "8 PPGs · 48 SKUs · mean conf 0.92"],
        ["ppg_selection", "done", "8/8 eligible"],
        ["feature_selection", "done", "8 candidates"],
        ["eda", "done", "top feature: price (ρ=-0.94)"],
        ["feature_engineering", "done", "800 rows · 16 features"],
        ["feature_refine", "done", "11 kept · max VIF 7.92"],
        ["modeling", "done", "stub"],
        ["results_reasoning", "done", "stub"],
      ].map(([name, status, summary], i) => (
        <div
          key={name}
          className="flex items-center gap-3 rounded border border-slate-800 bg-slate-900/40 px-3 py-2"
        >
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-400 text-[9px] font-bold text-slate-900">
            ✓
          </span>
          <span className="w-40 font-medium text-slate-100">{name}</span>
          <span className="rounded border border-emerald-500/40 bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">
            {status}
          </span>
          <span className="font-mono text-slate-400">{summary}</span>
        </div>
      ))}
    </div>
  );
}

export default function TabsMockup() {
  const [tab, setTab] = useState<Tab>("Data");
  return (
    <main>
      <div className="mb-4 rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-xs text-amber-200">
        <strong>Mockup · TAB layout.</strong> Pipeline stays narrow (one row per agent). Sister tabs
        hold the deep dives: <em>Data</em> (preview / schema / coverage / quality / anomalies),{" "}
        <em>Quality</em>, <em>PPG</em>, <em>Features</em>. Currently showing the <em>Data</em> tab.
      </div>
      <div className="mb-6 rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs uppercase tracking-wider text-slate-500">Run</p>
            <h1 className="font-mono text-lg">demo-run-7181d48</h1>
          </div>
          <span className="rounded-full border border-emerald-500/40 bg-emerald-500/15 px-3 py-1 text-xs text-emerald-300">
            completed · 14/14 · 12.4 s
          </span>
        </div>
        <div className="mt-5 flex gap-1 border-b border-slate-800">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`-mb-px border-b-2 px-3 py-2 text-xs font-medium transition ${
                t === tab
                  ? "border-emerald-400 text-emerald-300"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {tab === "Pipeline" && <Pipeline />}
      {tab === "Data" && (
        <div className="grid gap-4 md:grid-cols-2">
          <Section title="main.panel preview" span={2}>
            <DataPreview />
          </Section>
          <Section title="Schema">
            <SchemaTable />
          </Section>
          <Section title="Coverage SKU × week (red = missing)">
            <CoverageGrid />
          </Section>
          <Section title="Weekly trend">
            <TrendChart />
          </Section>
          <Section title="Anomalies flagged by ingestion">
            <AnomalyTable />
          </Section>
        </div>
      )}
      {tab === "Quality" && (
        <div className="grid gap-4 md:grid-cols-2">
          <Section title="dbt + GE checks">
            <QualityPanel />
          </Section>
          <Section title="Anomaly table">
            <AnomalyTable />
          </Section>
          <Section title="Coverage SKU × week">
            <CoverageGrid />
          </Section>
        </div>
      )}
      {tab === "PPG" && (
        <div className="grid gap-4 md:grid-cols-2">
          <Section title="SKU scatter — price tier × log price">
            <PPGScatter />
          </Section>
          <Section title="Within-PPG price distribution">
            <BoxPlot />
          </Section>
        </div>
      )}
      {tab === "Features" && (
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
    </main>
  );
}
