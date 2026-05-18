"use client";

import { useEffect, useState } from "react";
import { getArtifact } from "@/lib/api";
import { CoverageHeatmap, type CoverageData } from "./charts/CoverageHeatmap";
import { TrendChart, type TrendData } from "./charts/TrendChart";
import {
  PPGScatter,
  type FacetData,
  type TierOrBehaviourData,
} from "./charts/PPGScatter";
import { PPGPriceBox, type PriceBoxData } from "./charts/PPGPriceBox";
import { EligibilityBars, type EligibilityData } from "./charts/EligibilityBars";
import { PPGTabs } from "./PPGTabs";
import { PPGTable } from "./PPGTable";
import { DataPreview, type ProfileBlob } from "./tables/DataPreview";
import { SchemaTable } from "./tables/SchemaTable";
import { QualityPanel, type QualityData } from "./tables/QualityPanel";
import { AnomalyTable, type FindingsBlob } from "./tables/AnomalyTable";
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
