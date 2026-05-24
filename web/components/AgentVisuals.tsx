"use client";

import { useEffect, useMemo, useState } from "react";
import { getArtifact } from "@/lib/api";
import { useGrainState } from "@/lib/useGrainState";
import { GrainChipRow } from "./GrainChipRow";
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
import { ConstraintBinding, type ConstraintBindingRow } from "./charts/ConstraintBinding";
import { DecompStackedArea, type DecompPPGBlob } from "./charts/DecompStackedArea";
import { FittedVsActual, type FittedVsActualRow } from "./charts/FittedVsActual";
import { ResidualHistogram, type ResidualRow } from "./charts/ResidualHistogram";
import { SimulationHeatmap, type SimulationGridBlob } from "./charts/SimulationHeatmap";
import { PPGTabs } from "./PPGTabs";
import { PPGTable } from "./PPGTable";
import { CandidatesTable, type CandidatesRow } from "./tables/CandidatesTable";
import {
  ModelingByStore,
  type CellRow,
  type PooledRow,
} from "./tables/ModelingByStore";
import { ModelingProgress } from "./ModelingProgress";
import { DataPreview, type ProfileBlob } from "./tables/DataPreview";
import { DropLog, KeptList, type DropLogData } from "./tables/DropLog";
import { RecommendationTable, type RecommendationRow } from "./tables/RecommendationTable";
import { SchemaTable } from "./tables/SchemaTable";
import { QualityPanel, type QualityData } from "./tables/QualityPanel";
import { AnomalyTable, type FindingsBlob } from "./tables/AnomalyTable";
import {
  TargetRelationship,
  type TargetRelationshipRow,
} from "./tables/TargetRelationship";
import { ValidationTable, type ValidationRow } from "./tables/ValidationTable";
import { InsightsSummary, type InsightsSummaryBlob } from "./tables/InsightsSummary";
import { ConstraintEditor, type ConstraintsBlob } from "./ConstraintEditor";
import type { AgentName, AgentState, RunEvent } from "@/lib/types";

interface Props {
  runId: string;
  agent: AgentName;
  ready: boolean;
  events: RunEvent[];
  agentState?: AgentState;
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
    case "advanced_eda":
      return <AdvancedEDAVisuals {...props} />;
    case "feature_engineering":
      return <FeatureEngineeringVisuals {...props} />;
    case "feature_refine":
      return <FeatureRefineVisuals {...props} />;
    case "modeling":
      return <ModelingVisuals {...props} />;
    case "decomposition":
      return <DecompositionVisuals {...props} />;
    case "simulation":
      return <SimulationVisuals {...props} />;
    case "optimization":
      return <OptimizationVisuals {...props} />;
    case "validation":
      return <ValidationVisuals {...props} />;
    case "insights":
      return <InsightsVisuals {...props} />;
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

interface AdvancedEDASummary {
  summary: {
    n_ppgs_top_k: number;
    n_anomalies: number;
    n_change_points: number;
    stationarity_pass_rate: number;
    abc_counts: Record<string, number>;
    anomaly_breakdown: Record<string, number>;
  };
  findings: string[];
  narrative: string;
}

function AdvancedEDAVisuals({ runId, ready }: Props) {
  const report = useArtifact<AdvancedEDASummary>(runId, "advanced_eda_report.json", ready);
  if (!report || "missing_columns" in report) return null;
  const s = report.summary;
  return (
    <div className="mt-4 space-y-4 border-t border-slate-800 pt-4">
      <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-4">
        <Kpi label="PPGs analysed" value={s.n_ppgs_top_k} />
        <Kpi
          label="Stationary"
          value={`${Math.round(s.stationarity_pass_rate * 100)}%`}
        />
        <Kpi label="Anomalies" value={s.n_anomalies} />
        <Kpi label="Change points" value={s.n_change_points} />
      </div>
      {report.findings?.length > 0 && (
        <Section title="Top findings">
          <ul className="space-y-1 text-[11px] text-slate-300">
            {report.findings.slice(0, 5).map((f, i) => (
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

function Kpi({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex flex-col gap-1 rounded border border-slate-800 bg-slate-900/40 px-3 py-2">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">{label}</span>
      <span className="font-mono text-base text-emerald-300">{value}</span>
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
    grain_unit?: string | null;
  })[];
  n_correct_sign: number;
  n_retries: number;
  n_total: number;
  n_robust_refit?: number;
  skip_reasons?: Record<string, number>;
  model_pool: string[];
}

interface ShapEntry {
  ppg_id: string;
  model: string;
  shap: SHAPSummary;
}

function ModelingVisuals({ runId, ready, events, agentState }: Props) {
  const grain = useGrainState(runId, ready);
  const results = useArtifact<ModelingResults>(runId, grain.nameFor("modeling_results.json"), ready);
  const shapBlob = useArtifact<ShapEntry[]>(runId, grain.nameFor("shap_per_ppg.json"), ready);
  const posterior = useArtifact<PosteriorBlob>(
    runId,
    grain.nameFor("hierarchical_posterior.json"),
    ready,
  );
  const fvaBlob = useArtifact<FittedVsActualRow[]>(
    runId,
    grain.nameFor("fitted_vs_actual.json"),
    ready,
  );
  const pooledBlob = useArtifact<PooledRow[]>(
    runId,
    grain.nameFor("elasticity_per_ppg_pooled.json"),
    ready,
  );
  const rows = useMemo<CandidatesRow[]>(() => {
    if (!results || "missing_columns" in results) return [];
    return results.per_ppg.filter((r) => r.attempts && r.attempts.length > 0);
  }, [results]);
  // A run is at a store-grain when any modelling row carries a grain_unit
  // other than null / undefined / "chain". The pooled artifact is also
  // emitted only at store-grain; combining both keeps us robust to an
  // out-of-date browser cache.
  const isStoreGrain = useMemo(() => {
    if (!results || "missing_columns" in results) return false;
    return results.per_ppg.some(
      (r) => r.grain_unit && r.grain_unit !== "chain",
    );
  }, [results]);
  const storeCells = useMemo<CellRow[]>(() => {
    if (!isStoreGrain) return [];
    return rows
      .filter((r) => (r as CandidatesRow & { grain_unit?: string | null }).grain_unit)
      .map((r) => r as CellRow);
  }, [rows, isStoreGrain]);
  const pooled = useMemo<PooledRow[]>(
    () => (Array.isArray(pooledBlob) ? pooledBlob : []),
    [pooledBlob],
  );
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
  const fvaMap = new Map<string, FittedVsActualRow>(
    Array.isArray(fvaBlob) ? fvaBlob.map((r) => [r.ppg_id, r]) : [],
  );
  const selectedShap = selected ? shapMap.get(selected) : undefined;
  const selectedFva = selected ? fvaMap.get(selected) : undefined;
  const hasPosterior = posterior && !("missing_columns" in posterior) && posterior.n_studies > 0;
  const distinctPpgs = isStoreGrain
    ? new Set(storeCells.map((c) => c.ppg_id)).size
    : rows.length;
  const grainBadge = isStoreGrain ? (
    <span className="ml-2 rounded border border-sky-500/40 bg-sky-500/10 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-sky-200">
      Hoch grain · per-store
    </span>
  ) : null;
  const cellLabel = isStoreGrain
    ? `${distinctPpgs} PPGs · ${storeCells.length} store cells`
    : `${rows.length} PPGs`;

  return (
    <div className="mt-4 space-y-5 border-t border-slate-800 pt-4">
      <ModelingProgress events={events} agentState={agentState} />
      <GrainChipRow state={grain} />
      <Section title={`Candidate fits per PPG (winners marked) · ${cellLabel}`}>
        {grainBadge && <div className="-mt-1 mb-2">{grainBadge}</div>}
        {isStoreGrain ? (
          storeCells.length > 0 ? (
            <ModelingByStore
              cells={storeCells}
              pooled={pooled}
              selectedPpg={selected}
              onSelectPpg={setSelected}
            />
          ) : (
            <p className="text-[11px] text-slate-500">No store cells to display.</p>
          )
        ) : rows.length > 0 ? (
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
      {selectedFva && (
        <Section
          title={`Fitted vs actual · ${selectedFva.ppg_id} · ${selectedFva.model}`}
        >
          <p className="mb-2 text-[10.5px] text-slate-500">
            Train (green) and test (yellow) cells against the y = x identity. A
            tight cluster around the diagonal is what you want; bowing away
            shows where the model under- or over-shoots.
          </p>
          <FittedVsActual data={selectedFva} />
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

function DecompositionVisuals({ runId, ready }: Props) {
  const grain = useGrainState(runId, ready);
  const blob = useArtifact<DecompPPGBlob[]>(
    runId,
    grain.nameFor("decomposition_per_ppg_week.json"),
    ready,
  );
  const list = useMemo(() => (Array.isArray(blob) ? blob : []), [blob]);
  const [selected, setSelected] = useState<string | null>(null);
  useEffect(() => {
    if (list.length && (!selected || !list.some((r) => r.ppg_id === selected))) {
      setSelected(list[0].ppg_id);
    }
  }, [list, selected]);
  if (!list.length) return null;
  const current = list.find((r) => r.ppg_id === selected) ?? list[0];
  return (
    <div className="mt-4 space-y-3 border-t border-slate-800 pt-4">
      <GrainChipRow state={grain} />
      <Section title="Due-to decomposition over time">
        <div className="mb-2 flex flex-wrap items-center gap-1">
          {list.map((r) => (
            <button
              key={r.ppg_id}
              type="button"
              onClick={() => setSelected(r.ppg_id)}
              className={`rounded border px-2 py-0.5 font-mono text-[10px] ${
                r.ppg_id === current.ppg_id
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                  : "border-slate-700 bg-slate-900/60 text-slate-300 hover:bg-slate-800"
              }`}
            >
              {r.ppg_id}
            </button>
          ))}
        </div>
        <p className="mb-2 text-[10.5px] text-slate-500">
          Stacked areas = ``base`` + each driver group's contribution to weekly
          units; the thin white line is the observed weekly volume. Where the
          stack and the line diverge is the residual the model couldn't explain.
        </p>
        <DecompStackedArea data={current} />
      </Section>
    </div>
  );
}

function SimulationVisuals({ runId, ready }: Props) {
  const grain = useGrainState(runId, ready);
  const blob = useArtifact<SimulationGridBlob[]>(
    runId,
    grain.nameFor("simulation_grid.json"),
    ready,
  );
  const list = useMemo(() => (Array.isArray(blob) ? blob : []), [blob]);
  const [selected, setSelected] = useState<string | null>(null);
  const [metric, setMetric] = useState<"revenue" | "margin" | "units">("revenue");
  useEffect(() => {
    if (list.length && (!selected || !list.some((r) => r.ppg_id === selected))) {
      setSelected(list[0].ppg_id);
    }
  }, [list, selected]);
  if (!list.length) return null;
  const current = list.find((r) => r.ppg_id === selected) ?? list[0];
  return (
    <div className="mt-4 space-y-3 border-t border-slate-800 pt-4">
      <GrainChipRow state={grain} />
      <Section title={`Price × promo grid · ${current.ppg_id} · ${current.model_kind}`}>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-1">
          <div className="flex flex-wrap items-center gap-1">
            {list.map((r) => (
              <button
                key={r.ppg_id}
                type="button"
                onClick={() => setSelected(r.ppg_id)}
                className={`rounded border px-2 py-0.5 font-mono text-[10px] ${
                  r.ppg_id === current.ppg_id
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                    : "border-slate-700 bg-slate-900/60 text-slate-300 hover:bg-slate-800"
                }`}
              >
                {r.ppg_id}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1">
            {(["revenue", "margin", "units"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMetric(m)}
                className={`rounded border px-2 py-0.5 text-[10px] uppercase tracking-wider ${
                  metric === m
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                    : "border-slate-700 bg-slate-900/60 text-slate-300 hover:bg-slate-800"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
        <SimulationHeatmap data={current} metric={metric} />
      </Section>
    </div>
  );
}

interface OptResultsRow {
  ppg_id: string;
  milp: {
    feasible_strict?: boolean;
    relaxed?: boolean;
    chosen_slacks?: Record<string, number>;
  };
}

function OptimizationVisuals({ runId, ready }: Props) {
  const grain = useGrainState(runId, ready);
  const rows = useArtifact<RecommendationRow[]>(
    runId,
    grain.nameFor("optimization_table.json"),
    ready,
  );
  const constraints = useArtifact<ConstraintsBlob>(
    runId,
    grain.nameFor("optimization_constraints.json"),
    ready,
  );
  const results = useArtifact<OptResultsRow[]>(
    runId,
    grain.nameFor("optimization_results.json"),
    ready,
  );
  if (!rows && !constraints) return null;
  const recos = Array.isArray(rows) ? rows : [];
  const c = constraints && !("missing_columns" in constraints) ? constraints : null;
  const bindingRows: ConstraintBindingRow[] = Array.isArray(results)
    ? results
        .filter((r) => r.milp?.chosen_slacks && Object.keys(r.milp.chosen_slacks).length > 0)
        .map((r) => ({
          ppg_id: r.ppg_id,
          slacks: r.milp.chosen_slacks ?? {},
          feasible_strict: Boolean(r.milp.feasible_strict),
          relaxed: Boolean(r.milp.relaxed),
        }))
    : [];
  return (
    <div className="mt-4 space-y-5 border-t border-slate-800 pt-4">
      <GrainChipRow state={grain} />
      <Section title={`Recommendations · ${recos.length} PPGs · objective=${c?.objective ?? "—"}`}>
        <RecommendationTable rows={recos} />
      </Section>
      {bindingRows.length > 0 && (
        <Section title="Constraint slack at the chosen cell">
          <p className="mb-2 text-[10.5px] text-slate-500">
            Positive bars = slack (the constraint isn't binding); near-zero =
            the optimiser stopped at that boundary; negative = the constraint
            was relaxed and the violation magnitude is reported.
          </p>
          <ConstraintBinding rows={bindingRows} />
        </Section>
      )}
      {c && grain.isPrimary && (
        <Section title="Constraint editor · solve with defaults, edit, re-solve, approve">
          <ConstraintEditor runId={runId} current={c} />
        </Section>
      )}
      {c && !grain.isPrimary && (
        <p className="text-[10.5px] text-slate-500">
          Constraint editor is locked to the primary grain — switch back to ★ to
          edit constraints and re-solve.
        </p>
      )}
    </div>
  );
}

function InsightsVisuals({ runId, ready, agentState }: Props) {
  const grain = useGrainState(runId, ready);
  const summary = useArtifact<InsightsSummaryBlob>(
    runId,
    grain.nameFor("insights_summary.json"),
    ready,
  );
  if (!summary || "missing_columns" in summary) return null;
  const hasPdf = Boolean(agentState?.artifacts?.some((a) => a.name === "report.pdf"));
  return (
    <div className="mt-4 border-t border-slate-800 pt-4">
      <GrainChipRow state={grain} />
      <Section title={`Executive summary${grain.isPrimary ? "" : ` · ${grain.selected}`}`}>
        <InsightsSummary runId={runId} data={summary} hasPdf={hasPdf} />
      </Section>
    </div>
  );
}

function ValidationVisuals({ runId, ready }: Props) {
  const grain = useGrainState(runId, ready);
  const rows = useArtifact<ValidationRow[]>(
    runId,
    grain.nameFor("validation_table.json"),
    ready,
  );
  const residuals = useArtifact<ResidualRow[]>(
    runId,
    grain.nameFor("validation_residuals.json"),
    ready,
  );
  const list = Array.isArray(rows) ? rows : [];
  const residualList = Array.isArray(residuals) ? residuals : [];
  const [selected, setSelected] = useState<string | null>(null);
  useEffect(() => {
    if (residualList.length && (!selected || !residualList.some((r) => r.ppg_id === selected))) {
      setSelected(residualList[0].ppg_id);
    }
  }, [residualList, selected]);
  if (!rows && !residuals) return null;
  const current = residualList.find((r) => r.ppg_id === selected) ?? residualList[0];
  return (
    <div className="mt-4 space-y-5 border-t border-slate-800 pt-4">
      <GrainChipRow state={grain} />
      <Section title={`Rolling-origin CV verdicts · ${list.length} PPGs`}>
        <ValidationTable rows={list} />
      </Section>
      {current && (
        <Section title={`Hold-out residuals · ${current.ppg_id}`}>
          <div className="mb-2 flex flex-wrap items-center gap-1">
            {residualList.map((r) => (
              <button
                key={r.ppg_id}
                type="button"
                onClick={() => setSelected(r.ppg_id)}
                className={`rounded border px-2 py-0.5 font-mono text-[10px] ${
                  r.ppg_id === current.ppg_id
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                    : "border-slate-700 bg-slate-900/60 text-slate-300 hover:bg-slate-800"
                }`}
              >
                {r.ppg_id}
              </button>
            ))}
          </div>
          <p className="mb-2 text-[10.5px] text-slate-500">
            Histogram of ``observed - predicted`` on log-units, pooled across
            every CV fold's hold-out window. Centred near zero with a tight
            spread = stable fit; long tails = a few weeks the model badly
            misjudges.
          </p>
          <ResidualHistogram data={current} />
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
