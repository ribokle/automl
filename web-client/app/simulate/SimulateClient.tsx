"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { ArrowRight, Check, X } from "lucide-react";

import { fmtUsd, fmtIntCompact, fmtPct, fmtPctRaw } from "@/lib/format";
import { Card, RaisedCard } from "@/components/shell/Card";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ElasticityChart } from "@/components/charts/ElasticityChart";
import { useAnimatedNumber } from "@/lib/hooks/useAnimatedNumber";
import { cn } from "@/lib/cn";
import type { ClientPayload } from "@/lib/types";

interface Props {
  payload: ClientPayload;
}

export function SimulateClient({ payload }: Props) {
  const [ppgId, setPpgId] = React.useState(payload.recommendations[0].ppg_id);
  const [metric, setMetric] = React.useState<"revenue" | "units">("revenue");
  const rec = payload.recommendations.find((r) => r.ppg_id === ppgId)!;
  const curve = payload.elasticity_by_ppg[ppgId];

  const [price, setPrice] = React.useState<number>(rec.proposed_price);
  const [marginFloor, setMarginFloor] = React.useState(0.18);
  const [compGap, setCompGap] = React.useState(0.1);

  React.useEffect(() => {
    setPrice(rec.proposed_price);
  }, [ppgId, rec.proposed_price]);

  const closest = curve.reduce((best, p) =>
    Math.abs(p.price - price) < Math.abs(best.price - price) ? p : best,
  );
  const baseUnits = curve.find((p) => Math.abs(p.price - rec.current_price) < 0.05)?.units ?? 5000;
  const baseRevenue = rec.current_price * baseUnits;

  const deltaPct = (price - rec.current_price) / rec.current_price;
  const unitLift = (closest.units - baseUnits) / baseUnits;
  const margin = (price - rec.current_price * 0.6) / price;
  const compCeiling = rec.current_price * (1 + compGap);
  const ladderHi = rec.current_price * 1.15;
  const ladderLo = rec.current_price * 0.85;

  const animatedRevenue = useAnimatedNumber(closest.revenue, (v) => fmtUsd(v));
  const animatedUnits = useAnimatedNumber(closest.units, (v) => fmtIntCompact(v));
  const animatedDelta = useAnimatedNumber(deltaPct, (v) => fmtPct(v));
  const animatedLift = useAnimatedNumber(unitLift, (v) => fmtPct(v));

  const flags = [
    { ok: margin >= marginFloor, label: `Margin ${(margin * 100).toFixed(1)}% / floor ${(marginFloor * 100).toFixed(0)}%` },
    { ok: price <= compCeiling, label: `Within +${(compGap * 100).toFixed(0)}% competitor gap` },
    {
      ok: price >= ladderLo && price <= ladderHi,
      label: "Inside ±15% price ladder",
    },
  ];

  const baseRev = baseRevenue;

  return (
    <div className="grid gap-8 lg:grid-cols-[280px_1fr]">
      {/* Left rail: sliders */}
      <aside className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
            Scenario for
          </div>
          <Select value={ppgId} onValueChange={setPpgId}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {payload.recommendations.map((r) => (
                <SelectItem key={r.ppg_id} value={r.ppg_id}>
                  {r.ppg_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <SliderBlock
          label="Price"
          value={fmtUsd(price)}
          subValue={`current ${fmtUsd(rec.current_price)}`}
          min={Math.round(rec.current_price * 80) / 100}
          max={Math.round(rec.current_price * 120) / 100}
          step={0.01}
          current={price}
          onChange={(v) => setPrice(Number(v.toFixed(2)))}
          minLabel={fmtUsd(rec.current_price * 0.8)}
          maxLabel={fmtUsd(rec.current_price * 1.2)}
        />

        <SliderBlock
          label="Margin floor"
          value={`${(marginFloor * 100).toFixed(0)}%`}
          subValue="cuts blocked below this"
          min={0.1}
          max={0.35}
          step={0.01}
          current={marginFloor}
          onChange={setMarginFloor}
          minLabel="10%"
          maxLabel="35%"
        />

        <SliderBlock
          label="Competitor gap"
          value={`+${(compGap * 100).toFixed(0)}%`}
          subValue="upper bound vs comp price"
          min={0.05}
          max={0.25}
          step={0.01}
          current={compGap}
          onChange={setCompGap}
          minLabel="+5%"
          maxLabel="+25%"
        />
      </aside>

      {/* Right pane: KPIs + compare */}
      <div className="flex flex-col gap-8">
        <div className="grid grid-cols-2 gap-x-12 gap-y-6 md:grid-cols-4">
          <KpiLive label="Δ price" value={animatedDelta} positive={deltaPct >= 0} />
          <KpiLive label="Δ units" value={animatedLift} positive={unitLift >= 0} />
          <KpiLive label="Forecast units / week" value={animatedUnits} />
          <KpiLive
            label="Forecast revenue / week"
            value={animatedRevenue}
            positive={closest.revenue >= baseRev}
          />
        </div>

        <RaisedCard className="p-6">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
            <div>
              <div className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
                Price–demand curve
              </div>
              <h2 className="display mt-1 text-xl font-semibold">{rec.ppg_name}</h2>
              <p className="text-sm text-muted-foreground">
                Elasticity {rec.elasticity.toFixed(2)} · current {fmtUsd(rec.current_price)} ·
                recommended {fmtUsd(rec.proposed_price)}
              </p>
            </div>
            <Tabs value={metric} onValueChange={(m) => setMetric(m as "revenue" | "units")}>
              <TabsList>
                <TabsTrigger value="revenue">Revenue</TabsTrigger>
                <TabsTrigger value="units">Units</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
          <ElasticityChart
            data={curve}
            currentPrice={rec.current_price}
            proposedPrice={price}
            metric={metric}
            height={260}
          />
        </RaisedCard>

        <div className="grid gap-4 md:grid-cols-2">
          <ScenarioPanel
            title="Baseline"
            subtitle="Today's price"
            price={rec.current_price}
            revenue={baseRev}
            units={baseUnits}
          />
          <ScenarioPanel
            title="Your scenario"
            subtitle={
              <span className="inline-flex items-center gap-1 text-accent">
                <ArrowRight className="size-3" />
                live recompute
              </span>
            }
            price={price}
            revenue={closest.revenue}
            units={closest.units}
            highlight
          />
        </div>

        <Card className="flex flex-col gap-3 p-6">
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
            Guardrails
          </div>
          <ul className="flex flex-col gap-2">
            {flags.map((f) => (
              <motion.li
                key={f.label}
                initial={false}
                animate={{ x: 0 }}
                className="flex items-center justify-between gap-3 text-sm"
              >
                <span className={f.ok ? "text-foreground" : "text-negative"}>{f.label}</span>
                <span
                  className={cn(
                    "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium",
                    f.ok
                      ? "border-positive/30 bg-positive/10 text-positive"
                      : "border-negative/30 bg-negative/10 text-negative",
                  )}
                >
                  {f.ok ? <Check className="size-3" /> : <X className="size-3" />}
                  {f.ok ? "ok" : "breach"}
                </span>
              </motion.li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}

function SliderBlock({
  label,
  value,
  subValue,
  min,
  max,
  step,
  current,
  onChange,
  minLabel,
  maxLabel,
}: {
  label: string;
  value: string;
  subValue: string;
  min: number;
  max: number;
  step: number;
  current: number;
  onChange: (v: number) => void;
  minLabel: string;
  maxLabel: string;
}) {
  return (
    <div className="flex flex-col gap-2 border-t border-hairline pt-5 first-of-type:border-t-0 first-of-type:pt-0">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
          {label}
        </span>
        <span className="display text-xl font-semibold tabular">{value}</span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[current]}
        onValueChange={([v]) => onChange(v)}
      />
      <div className="flex justify-between text-[11px] tabular text-muted-foreground">
        <span>{minLabel}</span>
        <span>{subValue}</span>
        <span>{maxLabel}</span>
      </div>
    </div>
  );
}

function KpiLive({
  label,
  value,
  positive,
}: {
  label: string;
  value: string;
  positive?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </span>
      <span
        className={cn(
          "display text-2xl font-semibold tabular md:text-3xl",
          positive === undefined
            ? "text-foreground"
            : positive
              ? "text-positive"
              : "text-negative",
        )}
      >
        {value}
      </span>
    </div>
  );
}

function ScenarioPanel({
  title,
  subtitle,
  price,
  revenue,
  units,
  highlight,
}: {
  title: string;
  subtitle: React.ReactNode;
  price: number;
  revenue: number;
  units: number;
  highlight?: boolean;
}) {
  const animatedPrice = useAnimatedNumber(price, (v) => fmtUsd(v));
  const animatedRev = useAnimatedNumber(revenue, (v) => fmtUsd(v));
  const animatedUnits = useAnimatedNumber(units, (v) => fmtIntCompact(v));

  return (
    <Card
      className={cn(
        "p-6",
        highlight
          ? "ring-1 ring-accent/30 shadow-glow"
          : null,
      )}
    >
      <div className="flex items-baseline justify-between">
        <h3 className="font-display text-base font-semibold">{title}</h3>
        <span className="text-[11px] text-muted-foreground">{subtitle}</span>
      </div>
      <dl className="mt-5 grid grid-cols-3 gap-4">
        <ScenarioStat label="Price" value={animatedPrice} />
        <ScenarioStat label="Units / wk" value={animatedUnits} />
        <ScenarioStat label="Revenue / wk" value={animatedRev} primary />
      </dl>
    </Card>
  );
}

function ScenarioStat({
  label,
  value,
  primary,
}: {
  label: string;
  value: string;
  primary?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </dt>
      <dd
        className={cn(
          "tabular font-semibold",
          primary ? "display text-xl" : "display text-lg",
        )}
      >
        {value}
      </dd>
    </div>
  );
}
