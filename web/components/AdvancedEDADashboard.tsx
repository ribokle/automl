"use client";

import { useEffect, useMemo, useState } from "react";
import { getArtifact } from "@/lib/api";
import { ACFPlot, type ACFData } from "./charts/ACFPlot";
import { AnomalyTimeline, type AnomalyTimelineData } from "./charts/AnomalyTimeline";
import { CorrHeatmap, type CorrData } from "./charts/CorrHeatmap";
import { LorenzCurve, type LorenzData } from "./charts/LorenzCurve";
import { PriceLadderScatter, type PriceLadderData } from "./charts/PriceLadderScatter";
import { PromoCalendarHeatmap, type PromoCalendarData } from "./charts/PromoCalendarHeatmap";
import { PromoLiftBars, type PromoLiftRow } from "./charts/PromoLiftBars";
import { STLDecomposition, type STLData } from "./charts/STLDecomposition";
import { HolidayLiftTable, type HolidayLiftRow } from "./tables/HolidayLiftTable";

interface AdvancedEDAReport {
  summary: {
    n_ppgs_total: number;
    n_ppgs_top_k: number;
    stationarity_pass_rate: number;
    n_anomalies: number;
    anomaly_breakdown: Record<string, number>;
    n_change_points: number;
    abc_counts: Record<string, number>;
    top_lifters: Array<{ ppg_id: string; lift: number }>;
    categories_with_correlation: string[];
  };
  findings: string[];
  narrative: string;
  compute_caps: { max_series: number; corr_cap: number };
}

interface DiagnosticsBlob {
  max_series: number;
  n_ppgs_top_k: number;
  diagnostics: Array<{
    ppg_id: string;
    n_weeks: number;
    stl: STLData;
    acf_pacf: ACFData;
    stationarity: { verdict: string; adf_p: number | null; kpss_p: number | null };
  }>;
  store_variability: Array<{
    ppg_id: string;
    n_stores: number;
    store_cv_median: number;
    store_cv_max: number;
    zero_week_share: number;
  }>;
}

interface AnomaliesBlob {
  n_total: number;
  by_type: Record<string, number>;
  rows: Array<{
    ppg_id: string;
    week_start: string;
    anomaly_type: string;
    severity: number;
    note?: string;
  }>;
}

interface ChangePointsBlob {
  n_total: number;
  rows: Array<{
    ppg_id: string;
    week_start: string;
    pre_base_price: number;
    post_base_price: number;
    delta_pct: number;
  }>;
}

interface DistributionBlob {
  columns: Array<{
    column: string;
    n: number;
    skew: number;
    kurtosis: number;
    shapiro_p: number | null;
    iqr: number;
  }>;
}

interface PromoLiftBlob {
  per_ppg: Array<
    PromoLiftRow & {
      promo_window: {
        modal_length: number;
        mean_length: number;
        max_length: number;
        n_windows: number;
        pct_multiweek: number;
      };
    }
  >;
}

interface CrossPPGBlob {
  categories: Array<{ category: string; labels: string[]; matrix: number[][] }>;
}

interface ParetoBlob {
  sku: { volume?: LorenzData["volume"]; revenue?: LorenzData["revenue"] };
  brand: { volume?: LorenzData["volume"]; revenue?: LorenzData["revenue"] };
  store: { volume?: LorenzData["volume"]; revenue?: LorenzData["revenue"] };
  ppg_abc: Array<{ ppg_id: string; revenue: number; cum_share: number; abc_class: "A" | "B" | "C" }>;
}

interface PriceLadderBlob {
  per_ppg: PriceLadderData[];
}

interface HolidayBlob {
  rows: HolidayLiftRow[];
}

interface CardinalityBlob {
  columns: Array<{
    column: string;
    n_distinct: number;
    values: Array<{ value: string; n: number; share: number; rare: boolean }>;
  }>;
}

type Loaded<T> = T | null;

function useArtifact<T>(runId: string, name: string, enabled: boolean): Loaded<T> {
  const [data, setData] = useState<Loaded<T>>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    getArtifact<T>(runId, name)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [runId, name, enabled]);
  return data;
}

const SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "time-series", label: "Time series" },
  { id: "anomalies", label: "Anomalies" },
  { id: "change-points", label: "Change points" },
  { id: "distributions", label: "Distributions" },
  { id: "promo-lift", label: "Promo lift" },
  { id: "cross-ppg", label: "Cross-PPG" },
  { id: "pareto", label: "Pareto / ABC" },
  { id: "price-ladder", label: "Price ladder" },
  { id: "promo-calendar", label: "Promo calendar" },
  { id: "holiday", label: "Holiday lift" },
  { id: "cardinality", label: "Cardinality" },
] as const;

function KPI({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="flex flex-col gap-1 rounded border border-slate-800 bg-slate-900/40 px-3 py-2">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">{label}</span>
      <span className="font-mono text-lg text-emerald-300">{value}</span>
      {hint && <span className="text-[10px] text-slate-500">{hint}</span>}
    </div>
  );
}

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24 space-y-3 border-t border-slate-800 pt-6">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">{title}</h3>
      {children}
    </section>
  );
}

export function AdvancedEDADashboard({ runId }: { runId: string }) {
  const report = useArtifact<AdvancedEDAReport>(runId, "advanced_eda_report.json", true);
  const diag = useArtifact<DiagnosticsBlob>(runId, "time_series_diagnostics.json", true);
  const anom = useArtifact<AnomaliesBlob>(runId, "temporal_anomalies.json", true);
  const cps = useArtifact<ChangePointsBlob>(runId, "change_points.json", true);
  const dist = useArtifact<DistributionBlob>(runId, "distribution_report.json", true);
  const lift = useArtifact<PromoLiftBlob>(runId, "promo_lift_sketches.json", true);
  const cross = useArtifact<CrossPPGBlob>(runId, "cross_ppg_correlation.json", true);
  const pareto = useArtifact<ParetoBlob>(runId, "pareto_abc.json", true);
  const ladder = useArtifact<PriceLadderBlob>(runId, "price_ladder.json", true);
  const calendar = useArtifact<PromoCalendarData>(runId, "promo_calendar.json", true);
  const holiday = useArtifact<HolidayBlob>(runId, "holiday_lift.json", true);
  const cardinality = useArtifact<CardinalityBlob>(runId, "cardinality_report.json", true);
  const charts = useArtifact<{ anomaly_timeline: AnomalyTimelineData }>(runId, "advanced_eda_charts.json", true);

  const ppgList = useMemo(() => diag?.diagnostics.map((d) => d.ppg_id) ?? [], [diag]);
  const [tsPpg, setTsPpg] = useState<string | null>(null);
  const [anomPpg, setAnomPpg] = useState<string | null>(null);
  const [ladderPpg, setLadderPpg] = useState<string | null>(null);
  useEffect(() => {
    if (ppgList.length && !tsPpg) setTsPpg(ppgList[0]);
    if (ppgList.length && !anomPpg) setAnomPpg(ppgList[0]);
  }, [ppgList, tsPpg, anomPpg]);
  const ladderPpgList = useMemo(() => ladder?.per_ppg.map((p) => p.ppg_id) ?? [], [ladder]);
  useEffect(() => {
    if (ladderPpgList.length && !ladderPpg) setLadderPpg(ladderPpgList[0]);
  }, [ladderPpgList, ladderPpg]);

  if (!report) {
    return (
      <div className="rounded border border-slate-800 bg-slate-900/40 px-4 py-8 text-center text-[12px] text-slate-400">
        Advanced EDA artefacts not found for this run. Wait for the
        <code className="mx-1 rounded bg-slate-800 px-1 font-mono text-[10px]">advanced_eda</code>
        agent to finish, or rerun the pipeline.
      </div>
    );
  }

  const currentDiag = diag?.diagnostics.find((d) => d.ppg_id === tsPpg);
  const currentLadder = ladder?.per_ppg.find((p) => p.ppg_id === ladderPpg);

  return (
    <div className="flex gap-6">
      <nav className="sticky top-20 hidden h-fit w-44 shrink-0 flex-col gap-1 lg:flex">
        {SECTIONS.map((s) => (
          <a
            key={s.id}
            href={`#${s.id}`}
            className="rounded px-2 py-1 text-[11px] text-slate-400 hover:bg-slate-900/60 hover:text-emerald-300"
          >
            {s.label}
          </a>
        ))}
      </nav>
      <div className="min-w-0 flex-1 space-y-6">
        <Section id="overview" title="Overview">
          <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
            <KPI label="PPGs analysed" value={report.summary.n_ppgs_top_k} hint={`of ${report.summary.n_ppgs_total} total`} />
            <KPI
              label="Stationary"
              value={`${Math.round(report.summary.stationarity_pass_rate * 100)}%`}
              hint="ADF rejects + KPSS doesn't"
            />
            <KPI label="Anomalies" value={report.summary.n_anomalies} hint={Object.entries(report.summary.anomaly_breakdown).map(([k, v]) => `${k}:${v}`).join(" · ")} />
            <KPI label="Change points" value={report.summary.n_change_points} hint="baseline-price shifts" />
            <KPI label="ABC=A" value={report.summary.abc_counts.A ?? 0} hint="PPGs in top 80% of revenue" />
            <KPI label="Categories" value={report.summary.categories_with_correlation.length} hint="with intra-cat corr" />
          </div>
          {report.findings.length > 0 && (
            <div>
              <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">Findings</h4>
              <ul className="space-y-1 text-[11px] text-slate-300">
                {report.findings.map((f, i) => (
                  <li key={i} className="rounded border border-slate-800 bg-slate-900/40 px-2 py-1">
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {report.narrative && (
            <p className="text-[11px] italic text-slate-400">{report.narrative}</p>
          )}
        </Section>

        <Section id="time-series" title="Time-series diagnostics">
          {ppgList.length > 0 && (
            <div className="flex flex-wrap items-center gap-1">
              {ppgList.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setTsPpg(p)}
                  className={`rounded border px-2 py-0.5 font-mono text-[10px] ${
                    p === tsPpg
                      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                      : "border-slate-700 bg-slate-900/60 text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  {p}
                </button>
              ))}
            </div>
          )}
          {currentDiag && (
            <>
              <div className="flex flex-wrap items-center gap-2 text-[11px]">
                <span className="rounded border border-slate-700 bg-slate-900/60 px-2 py-0.5 text-slate-300">
                  verdict: <b className="text-emerald-300">{currentDiag.stationarity.verdict}</b>
                </span>
                {currentDiag.stationarity.adf_p !== null && (
                  <span className="text-slate-500">ADF p={currentDiag.stationarity.adf_p.toFixed(3)}</span>
                )}
                {currentDiag.stationarity.kpss_p !== null && (
                  <span className="text-slate-500">KPSS p={currentDiag.stationarity.kpss_p.toFixed(3)}</span>
                )}
              </div>
              <STLDecomposition data={currentDiag.stl} />
              <ACFPlot data={currentDiag.acf_pacf} />
            </>
          )}
        </Section>

        <Section id="anomalies" title="Temporal anomalies">
          {anom && (
            <div className="flex flex-wrap gap-2 text-[11px]">
              {Object.entries(anom.by_type).map(([t, n]) => (
                <span key={t} className="rounded border border-slate-700 bg-slate-900/60 px-2 py-0.5 text-slate-300">
                  {t}: <b className="text-emerald-300">{n}</b>
                </span>
              ))}
            </div>
          )}
          {anomPpg && ppgList.length > 0 && (
            <div className="flex flex-wrap items-center gap-1">
              {ppgList.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setAnomPpg(p)}
                  className={`rounded border px-2 py-0.5 font-mono text-[10px] ${
                    p === anomPpg
                      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                      : "border-slate-700 bg-slate-900/60 text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  {p}
                </button>
              ))}
            </div>
          )}
          {charts && anomPpg && (
            <AnomalyTimeline data={charts.anomaly_timeline} ppg={anomPpg} />
          )}
        </Section>

        <Section id="change-points" title="Baseline-price change points">
          {cps && cps.rows.length > 0 ? (
            <div className="max-h-72 overflow-y-auto rounded border border-slate-800">
              <table className="w-full text-[11px]">
                <thead className="sticky top-0 bg-slate-900 text-slate-400">
                  <tr>
                    <th className="px-2 py-1 text-left">PPG</th>
                    <th className="px-2 py-1 text-left">Week</th>
                    <th className="px-2 py-1 text-right">Pre</th>
                    <th className="px-2 py-1 text-right">Post</th>
                    <th className="px-2 py-1 text-right">Δ%</th>
                  </tr>
                </thead>
                <tbody>
                  {cps.rows.map((r, i) => (
                    <tr key={i} className="border-t border-slate-800">
                      <td className="px-2 py-1 font-mono">{r.ppg_id}</td>
                      <td className="px-2 py-1 text-slate-400">{r.week_start.split("T")[0]}</td>
                      <td className="px-2 py-1 text-right">${r.pre_base_price.toFixed(2)}</td>
                      <td className="px-2 py-1 text-right">${r.post_base_price.toFixed(2)}</td>
                      <td className={`px-2 py-1 text-right ${r.delta_pct > 0 ? "text-amber-300" : "text-rose-300"}`}>
                        {r.delta_pct >= 0 ? "+" : ""}
                        {r.delta_pct.toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-[11px] text-slate-500">No change points detected.</p>
          )}
        </Section>

        <Section id="distributions" title="Numeric distributions">
          {dist && dist.columns.length > 0 ? (
            <div className="overflow-x-auto rounded border border-slate-800">
              <table className="w-full text-[11px]">
                <thead className="bg-slate-900 text-slate-400">
                  <tr>
                    <th className="px-2 py-1 text-left">Column</th>
                    <th className="px-2 py-1 text-right">n</th>
                    <th className="px-2 py-1 text-right">Skew</th>
                    <th className="px-2 py-1 text-right">Kurt</th>
                    <th className="px-2 py-1 text-right">IQR</th>
                    <th className="px-2 py-1 text-right">Shapiro p</th>
                  </tr>
                </thead>
                <tbody>
                  {dist.columns.map((c) => (
                    <tr key={c.column} className="border-t border-slate-800">
                      <td className="px-2 py-1 font-mono">{c.column}</td>
                      <td className="px-2 py-1 text-right text-slate-400">{c.n}</td>
                      <td className="px-2 py-1 text-right">{c.skew.toFixed(2)}</td>
                      <td className="px-2 py-1 text-right">{c.kurtosis.toFixed(2)}</td>
                      <td className="px-2 py-1 text-right">{c.iqr.toFixed(3)}</td>
                      <td className="px-2 py-1 text-right">
                        {c.shapiro_p === null ? "—" : c.shapiro_p < 0.05 ? (
                          <span className="text-amber-300">{c.shapiro_p.toFixed(3)}</span>
                        ) : (
                          <span className="text-slate-300">{c.shapiro_p.toFixed(3)}</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-[11px] text-slate-500">No distributions to display.</p>
          )}
        </Section>

        <Section id="promo-lift" title="Promo lift sketches (pre-model)">
          <p className="text-[10.5px] italic text-slate-500">
            Median-split of weekly promo share per PPG; lift = on / off mean
            units. Pre-model — the modelling agent recovers the controlled
            elasticity.
          </p>
          {lift && lift.per_ppg.length > 0 && (
            <>
              <PromoLiftBars rows={lift.per_ppg} />
              <div className="overflow-x-auto rounded border border-slate-800">
                <table className="w-full text-[11px]">
                  <thead className="bg-slate-900 text-slate-400">
                    <tr>
                      <th className="px-2 py-1 text-left">PPG</th>
                      <th className="px-2 py-1 text-right">Modal window</th>
                      <th className="px-2 py-1 text-right">Mean</th>
                      <th className="px-2 py-1 text-right">Max</th>
                      <th className="px-2 py-1 text-right">% multi-week</th>
                    </tr>
                  </thead>
                  <tbody>
                    {lift.per_ppg.map((r) => (
                      <tr key={r.ppg_id} className="border-t border-slate-800">
                        <td className="px-2 py-1 font-mono">{r.ppg_id}</td>
                        <td className="px-2 py-1 text-right">{r.promo_window.modal_length}</td>
                        <td className="px-2 py-1 text-right">{r.promo_window.mean_length.toFixed(1)}</td>
                        <td className="px-2 py-1 text-right">{r.promo_window.max_length}</td>
                        <td className="px-2 py-1 text-right">{(r.promo_window.pct_multiweek * 100).toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </Section>

        <Section id="cross-ppg" title="Cross-PPG demand correlation (within category)">
          <p className="text-[10.5px] italic text-slate-500">
            Uncontrolled correlation of detrended weekly units. Negative values
            *may* indicate substitution, but common-cause drivers (holiday,
            weather) are also possible — interpret as a coherence check, not
            cannibalisation evidence.
          </p>
          {cross && cross.categories.length > 0 ? (
            cross.categories.map((c) => (
              <div key={c.category} className="space-y-1">
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">{c.category}</h4>
                <CorrHeatmap data={{ labels: c.labels, matrix: c.matrix } as CorrData} />
              </div>
            ))
          ) : (
            <p className="text-[11px] text-slate-500">No multi-PPG categories.</p>
          )}
        </Section>

        <Section id="pareto" title="Pareto / ABC">
          {pareto && (
            <>
              <div className="grid gap-4 md:grid-cols-3">
                {pareto.sku?.revenue && (
                  <LorenzCurve
                    data={{
                      dimension: "sku",
                      volume: pareto.sku.volume ?? { x: [], cum_share: [], labels: [] },
                      revenue: pareto.sku.revenue ?? { x: [], cum_share: [], labels: [] },
                    }}
                  />
                )}
                {pareto.brand?.revenue && (
                  <LorenzCurve
                    data={{
                      dimension: "brand",
                      volume: pareto.brand.volume ?? { x: [], cum_share: [], labels: [] },
                      revenue: pareto.brand.revenue ?? { x: [], cum_share: [], labels: [] },
                    }}
                  />
                )}
                {pareto.store?.revenue && (
                  <LorenzCurve
                    data={{
                      dimension: "store",
                      volume: pareto.store.volume ?? { x: [], cum_share: [], labels: [] },
                      revenue: pareto.store.revenue ?? { x: [], cum_share: [], labels: [] },
                    }}
                  />
                )}
              </div>
              <div className="overflow-x-auto rounded border border-slate-800">
                <table className="w-full text-[11px]">
                  <thead className="bg-slate-900 text-slate-400">
                    <tr>
                      <th className="px-2 py-1 text-left">PPG</th>
                      <th className="px-2 py-1 text-right">Revenue</th>
                      <th className="px-2 py-1 text-right">Cum %</th>
                      <th className="px-2 py-1 text-left">ABC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pareto.ppg_abc.map((r) => {
                      const colour =
                        r.abc_class === "A"
                          ? "text-emerald-300"
                          : r.abc_class === "B"
                            ? "text-amber-300"
                            : "text-slate-400";
                      return (
                        <tr key={r.ppg_id} className="border-t border-slate-800">
                          <td className="px-2 py-1 font-mono">{r.ppg_id}</td>
                          <td className="px-2 py-1 text-right">${r.revenue.toLocaleString()}</td>
                          <td className="px-2 py-1 text-right">{(r.cum_share * 100).toFixed(1)}%</td>
                          <td className={`px-2 py-1 font-semibold ${colour}`}>{r.abc_class}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </Section>

        <Section id="price-ladder" title="Price ladder">
          {ladderPpgList.length > 0 && (
            <div className="flex flex-wrap items-center gap-1">
              {ladderPpgList.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setLadderPpg(p)}
                  className={`rounded border px-2 py-0.5 font-mono text-[10px] ${
                    p === ladderPpg
                      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                      : "border-slate-700 bg-slate-900/60 text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  {p}
                </button>
              ))}
            </div>
          )}
          {currentLadder && (
            <>
              <div className="flex flex-wrap gap-2 text-[11px]">
                <span className="rounded border border-slate-700 bg-slate-900/60 px-2 py-0.5 text-slate-300">
                  distinct prices: <b>{currentLadder.ladder.length}</b>
                </span>
                {currentLadder.price_volume_slope !== null && (
                  <span className="rounded border border-slate-700 bg-slate-900/60 px-2 py-0.5 text-slate-300">
                    price-volume slope: <b className={currentLadder.price_volume_slope < 0 ? "text-emerald-300" : "text-rose-300"}>
                      {currentLadder.price_volume_slope.toFixed(2)}
                    </b>
                  </span>
                )}
              </div>
              {currentLadder.caveat && (
                <p className="text-[10.5px] italic text-amber-200/80">{currentLadder.caveat}</p>
              )}
              <PriceLadderScatter data={currentLadder} />
            </>
          )}
        </Section>

        <Section id="promo-calendar" title="Promo calendar (week × PPG)">
          {calendar && <PromoCalendarHeatmap data={calendar} />}
        </Section>

        <Section id="holiday" title="Holiday lift">
          {holiday && <HolidayLiftTable rows={holiday.rows} />}
        </Section>

        <Section id="cardinality" title="Categorical cardinality">
          {cardinality && cardinality.columns.length > 0 ? (
            <div className="grid gap-3 md:grid-cols-2">
              {cardinality.columns.map((c) => (
                <div key={c.column} className="rounded border border-slate-800 bg-slate-900/40 p-2">
                  <h4 className="mb-1 text-[11px] font-semibold text-slate-300">
                    {c.column}{" "}
                    <span className="font-mono text-slate-500">({c.n_distinct})</span>
                  </h4>
                  <table className="w-full text-[11px]">
                    <tbody>
                      {c.values.slice(0, 10).map((v) => (
                        <tr key={v.value} className="border-t border-slate-800">
                          <td className="px-2 py-1">{v.value}</td>
                          <td className="px-2 py-1 text-right text-slate-400">{v.n}</td>
                          <td className="px-2 py-1 text-right">
                            {v.rare ? (
                              <span className="text-amber-300">{(v.share * 100).toFixed(1)}%</span>
                            ) : (
                              <span className="text-slate-300">{(v.share * 100).toFixed(1)}%</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-[11px] text-slate-500">No categorical columns to report.</p>
          )}
        </Section>
      </div>
    </div>
  );
}
