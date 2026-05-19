"use client";

import { useEffect, useMemo, useState } from "react";
import { getArtifact } from "@/lib/api";
import { CorrHeatmap, type CorrData } from "./charts/CorrHeatmap";
import { CoverageHeatmap, type CoverageData } from "./charts/CoverageHeatmap";
import { FeatureHistograms, type HistogramsData } from "./charts/FeatureHistograms";
import { TrendChart, type TrendData } from "./charts/TrendChart";
import {
  PPGScatter,
  type FacetData,
  type TierOrBehaviourData,
} from "./charts/PPGScatter";
import { PPGPriceBox, type PriceBoxData } from "./charts/PPGPriceBox";
import { EligibilityBars, type EligibilityData } from "./charts/EligibilityBars";
import { VIFBar } from "./charts/VIFBar";
import { SHAPBar, type SHAPSummary } from "./charts/SHAPBar";
import { ElasticityForest, type PosteriorBlob } from "./charts/ElasticityForest";
import { PPGTabs } from "./PPGTabs";
import { PPGTable } from "./PPGTable";
import { CandidatesTable, type CandidatesRow } from "./tables/CandidatesTable";
import { DataPreview, type ProfileBlob } from "./tables/DataPreview";
import { DropLog, KeptList, type DropLogData } from "./tables/DropLog";
import { SchemaTable } from "./tables/SchemaTable";
import { QualityPanel, type QualityData } from "./tables/QualityPanel";
import { AnomalyTable, type FindingsBlob } from "./tables/AnomalyTable";
import {
  TargetRelationship,
  type TargetRelationshipRow,
} from "./tables/TargetRelationship";
import type { AgentName, RunEvent } from "@/lib/types";

interface Props {
  runId: string;
  agent: AgentName;
  ready: boolean;
  events: RunEvent[];
}

type Loaded<T> = T | null | { missing_columns: string[] };

function useArtifact<T>(runId: string, name: string, ready: boolean): Loaded<T> {
  const [data, setData] = useState<Loaded<T>>(null);
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    getArtifact<T>(runId, name)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [runId, name, ready]);
  return data;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">{title}</h4>
      {children}
    </section>
  );
}

export function AgentVisuals(props: Props) {
  switch (props.agent) {
    case "ingestion":
      return <IngestionVisuals {...props} />;
    case "ppg_mapping":
      return <PPGMappingVisuals {...props} />;
    case "ppg_selection":
      return <PPGSelectionVisuals {...props} />;
    case "eda":
      return <EDAVisuals {...props} />;
    case "feature_engineering":
      return <FeatureEngineeringVisuals {...props} />;
    case "feature_refine":
      return <FeatureRefineVisuals {...props} />;
    case "modeling":
      return <ModelingVisuals {...props} />;
    default:
      return null;
  }
}

function IngestionVisuals({ runId, ready }: Props) {
  const profile = useArtifact<ProfileBlob>(runId, "data_profile.json", ready);
  const quality = useArtifact<QualityData>(runId, "quality_results.json", ready);
  const findings = useArtifact<FindingsBlob>(runId, "ingestion_findings.json", ready);
  const coverage = useArtifact<CoverageData>(runId, "coverage_grid.json", ready);
  const trend = useArtifact<TrendData>(runId, "weekly_trend.json", ready);
  if (!profile && !quality && !findings && !coverage && !trend) return null;
  return (
    <div className="mt-4 grid gap-5 border-t border-slate-800 pt-4 md:grid-cols-2">
      {profile && !("missing_columns" in profile) && (
        <Section title="main.panel preview">
          <DataPreview data={profile} />
        </Section>
      )}
      {profile && !("missing_columns" in profile) && (
        <Section title="Schema · column roles">
          <SchemaTable data={profile} />
        </Section>
      )}
      {coverage && !("missing_columns" in coverage) && (
        <Section title={`Coverage · SKU × week (${coverage.n_present_cells}/${coverage.n_total_cells} cells)`}>
          <CoverageHeatmap data={coverage} />
        </Section>
      )}
      {trend && !("missing_columns" in trend) && (
        <Section title="Weekly units & average price">
          <TrendChart data={trend} />
        </Section>
      )}
      {quality && !("missing_columns" in quality) && (
        <Section title="Quality checks · dbt + Great Expectations">
          <QualityPanel data={quality} />
        </Section>
      )}
      {findings && !("missing_columns" in findings) && (
        <Section title="Anomalies & narrative">
          <AnomalyTable data={findings} />
        </Section>
      )}
    </div>
  );
}

function PPGMappingVisuals({ runId, ready, events }: Props) {
  const tier = useArtifact<TierOrBehaviourData>(runId, "ppg_scatter_tier.json", ready);
  const behaviour = useArtifact<TierOrBehaviourData>(runId, "ppg_scatter_behaviour.json", ready);
  const facet = useArtifact<FacetData>(runId, "ppg_scatter_facet.json", ready);
  const box = useArtifact<PriceBoxData>(runId, "ppg_price_box.json", ready);
  if (!tier && !behaviour && !facet && !box) return null;
  const tabs = [
    tier && {
      key: "tier",
      label: "Tier × log-price",
      description: "X = pack-size tier (small / medium / large), Y = log of median price. Confirms the price-pack partition.",
      content: <PPGScatter data={tier} />,
    },
    behaviour && {
      key: "behaviour",
      label: "Behaviour",
      description: "X = log mean weekly units per SKU, Y = per-SKU corr(log units, log price). Members of the same PPG should cluster.",
      content: <PPGScatter data={behaviour} />,
    },
    facet && {
      key: "facet",
      label: "Faceted brand × pack",
      description: "One panel per category; tests the brand / pack-size separation the clusterer was supposed to honour.",
      content: <PPGScatter data={facet} />,
    },
  ].filter(Boolean) as { key: string; label: string; description: string; content: React.ReactNode }[];
  return (
    <div className="mt-4 space-y-5 border-t border-slate-800 pt-4">
      <div className="grid gap-5 md:grid-cols-2">
        {tabs.length > 0 && (
          <Section title="SKU scatter">
            <PPGTabs tabs={tabs} />
          </Section>
        )}
        {box && (
          <Section title="Within-PPG price distribution">
            <PPGPriceBox data={box} />
          </Section>
        )}
      </div>
      <PPGTable runId={runId} events={events} />
    </div>
  );
}

function PPGSelectionVisuals({ runId, ready }: Props) {
  const bars = useArtifact<EligibilityData>(runId, "ppg_eligibility_bars.json", ready);
  if (!bars) return null;
  return (
    <div className="mt-4 border-t border-slate-800 pt-4">
      <Section title="Per-PPG eligibility (stacked contributions)">
        <EligibilityBars data={bars} />
      </Section>
    </div>
  );
}

interface EDAReport {
  target_relationship: TargetRelationshipRow[];
  findings: string[];
  narrative: string;
}

function EDAVisuals({ runId, ready }: Props) {
  const trend = useArtifact<TrendData>(runId, "weekly_trend.json", ready);
  const corr = useArtifact<CorrData>(runId, "eda_corr_matrix.json", ready);
  const report = useArtifact<EDAReport>(runId, "eda_report.json", ready);
  if (!trend && !corr && !report) return null;
  return (
    <div className="mt-4 grid gap-5 border-t border-slate-800 pt-4 md:grid-cols-2">
      {trend && !("missing_columns" in trend) && (
        <Section title="Weekly units & average price">
          <TrendChart data={trend} />
        </Section>
      )}
      {corr && !("missing_columns" in corr) && (
        <Section title="Pairwise correlation · numeric candidates">
          <CorrHeatmap data={corr} />
        </Section>
      )}
      {report && !("missing_columns" in report) && (
        <Section title="Target relationship · ranked by |spearman ρ|">
          <TargetRelationship rows={report.target_relationship} />
        </Section>
      )}
      {report && !("missing_columns" in report) && report.findings?.length > 0 && (
        <Section title="EDA findings">
          <ul className="space-y-1 text-[11px] text-slate-300">
            {report.findings.map((f, i) => (
              <li key={i} className="rounded border border-slate-800 bg-slate-900/40 px-2 py-1">
                {f}
              </li>
            ))}
          </ul>
        </Section>
      )}
    </div>
  );
}

function FeatureEngineeringVisuals({ runId, ready }: Props) {
  const hist = useArtifact<HistogramsData>(runId, "feature_histograms.json", ready);
  if (!hist) return null;
  return (
    <div className="mt-4 border-t border-slate-800 pt-4">
      <Section title="Engineered-feature distributions (20-bin histograms)">
        <FeatureHistograms data={hist} />
      </Section>
    </div>
  );
}

interface RefineReport {
  vif: Record<string, number>;
  kept: string[];
  dropped: { feature: string; reason: string }[];
  max_vif: number;
  max_abs_corr: number;
  vif_threshold: number;
  passes_thresholds: boolean;
}

interface ModelingResults {
  controls_used: string[];
  per_ppg: (CandidatesRow & {
    winner: { diagnostics: { shap?: SHAPSummary } } | null;
  })[];
  n_correct_sign: number;
  n_retries: number;
  n_total: number;
  model_pool: string[];
}

interface ShapEntry {
  ppg_id: string;
  model: string;
  shap: SHAPSummary;
}

function ModelingVisuals({ runId, ready }: Props) {
  const results = useArtifact<ModelingResults>(runId, "modeling_results.json", ready);
  const shapBlob = useArtifact<ShapEntry[]>(runId, "shap_per_ppg.json", ready);
  const posterior = useArtifact<PosteriorBlob>(runId, "hierarchical_posterior.json", ready);
  const rows = useMemo<CandidatesRow[]>(() => {
    if (!results || "missing_columns" in results) return [];
    return results.per_ppg.filter((r) => r.attempts && r.attempts.length > 0);
  }, [results]);
  const [selected, setSelected] = useState<string | null>(null);
  useEffect(() => {
    if (rows.length && (!selected || !rows.some((r) => r.ppg_id === selected))) {
      setSelected(rows[0].ppg_id);
    }
  }, [rows, selected]);
  if (!results && !shapBlob && !posterior) return null;
  const shapMap = new Map<string, ShapEntry>(
    Array.isArray(shapBlob) ? shapBlob.map((s) => [s.ppg_id, s]) : [],
  );
  const selectedShap = selected ? shapMap.get(selected) : undefined;
  const hasPosterior = posterior && !("missing_columns" in posterior) && posterior.n_studies > 0;
  return (
    <div className="mt-4 space-y-5 border-t border-slate-800 pt-4">
      <Section title={`Candidate fits per PPG (winners marked) · ${rows.length} PPGs`}>
        {rows.length > 0 ? (
          <CandidatesTable rows={rows} selectedPpg={selected} onSelectPpg={setSelected} />
        ) : (
          <p className="text-[11px] text-slate-500">No fits to display.</p>
        )}
      </Section>
      {hasPosterior && (
        <Section
          title={`Forest plot · empirical-Bayes shrinkage across ${posterior.n_studies} OLS winners (τ² = ${posterior.tau_squared.toFixed(3)})`}
        >
          <p className="mb-2 text-[10.5px] text-slate-500">
            Grey whiskers = per-PPG OLS point ± 95% CI. Green diamonds = posterior
            after partial-pooling toward the population mean (yellow dashed). PPGs
            with wider SE get pulled harder toward μ̂.
          </p>
          <ElasticityForest data={posterior} />
        </Section>
      )}
      {selectedShap && (
        <Section
          title={`Feature attribution · ${selectedShap.ppg_id} · ${selectedShap.shap.method === "tree_shap" ? "tree SHAP" : "centred OLS contribution"}`}
        >
          <p className="mb-2 text-[10.5px] text-slate-500">
            Bars show mean |SHAP| for the winner ({selectedShap.model}). Blue =
            feature pushes log-units down on average; green = pushes up.
          </p>
          <SHAPBar data={selectedShap.shap} />
        </Section>
      )}
    </div>
  );
}

function FeatureRefineVisuals({ runId, ready }: Props) {
  const report = useArtifact<RefineReport>(runId, "feature_refine.json", ready);
  const corr = useArtifact<CorrData>(runId, "corr_refined.json", ready);
  if (!report && !corr) return null;
  const dropLog: DropLogData | null = report && !("missing_columns" in report)
    ? { kept: report.kept, dropped: report.dropped }
    : null;
  return (
    <div className="mt-4 space-y-5 border-t border-slate-800 pt-4">
      <div className="grid gap-5 md:grid-cols-2">
        {report && !("missing_columns" in report) && (
          <Section title={`VIF per kept feature · threshold ${report.vif_threshold}`}>
            <VIFBar data={{ vif: report.vif, threshold: report.vif_threshold }} />
          </Section>
        )}
        {corr && !("missing_columns" in corr) && (
          <Section title="Pairwise correlation · refined set">
            <CorrHeatmap data={corr} />
          </Section>
        )}
      </div>
      {dropLog && (
        <div className="grid gap-5 md:grid-cols-2">
          <Section title={`Dropped (${dropLog.dropped.length})`}>
            <DropLog data={dropLog} />
          </Section>
          <Section title={`Kept (${dropLog.kept.length})`}>
            <KeptList data={dropLog} />
          </Section>
        </div>
      )}
    </div>
  );
}
