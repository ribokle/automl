"use client";

import * as React from "react";
import { motion } from "framer-motion";

import { fmtUsd, fmtIntCompact, fmtPct } from "@/lib/format";
import { Surface } from "@/components/shell/Surface";
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
import type { ClientPayload, Variant } from "@/lib/types";

interface Props {
  payload: ClientPayload;
  variant: Variant;
}

export function SimulateClient({ payload, variant }: Props) {
  const [ppgId, setPpgId] = React.useState(payload.recommendations[0].ppg_id);
  const [metric, setMetric] = React.useState<"revenue" | "units">("revenue");
  const rec = payload.recommendations.find((r) => r.ppg_id === ppgId)!;
  const curve = payload.elasticity_by_ppg[ppgId];
  const [price, setPrice] = React.useState<number>(rec.proposed_price);

  React.useEffect(() => {
    setPrice(rec.proposed_price);
  }, [ppgId, rec.proposed_price]);

  const closest = curve.reduce((best, p) =>
    Math.abs(p.price - price) < Math.abs(best.price - price) ? p : best,
  );
  const baseRevenue = rec.current_price * 5000;
  const deltaPct = (price - rec.current_price) / rec.current_price;
  const unitLift = -rec.elasticity * deltaPct;
  const margin = (price - rec.current_price * 0.6) / price;
  const marginFloor = 0.18;
  const compGap = rec.current_price * 1.1;
  const ladderHi = rec.current_price * 1.15;
  const ladderLo = rec.current_price * 0.85;

  const flags: Array<{ ok: boolean; label: string }> = [
    { ok: margin >= marginFloor, label: `Margin ${(margin * 100).toFixed(1)}% / floor 18%` },
    { ok: price <= compGap, label: `Within +10% competitor gap` },
    {
      ok: price >= ladderLo && price <= ladderHi,
      label: "Inside ±15% price ladder",
    },
  ];

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <Surface variant={variant} tone="raised" className="lg:col-span-2 flex flex-col gap-4 p-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <div className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
              Price–demand curve
            </div>
            <h2 className="font-display text-xl font-semibold">{rec.ppg_name}</h2>
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
          variant={variant}
          data={curve}
          currentPrice={rec.current_price}
          proposedPrice={price}
          metric={metric}
        />
        <div className="mt-2 flex flex-col gap-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Test price</span>
            <span className="font-display text-lg font-semibold tabular">{fmtUsd(price)}</span>
          </div>
          <Slider
            min={Math.round(rec.current_price * 80) / 100}
            max={Math.round(rec.current_price * 120) / 100}
            step={0.01}
            value={[price]}
            onValueChange={([v]) => setPrice(Number(v.toFixed(2)))}
          />
          <div className="flex justify-between text-xs text-muted-foreground tabular">
            <span>{fmtUsd(rec.current_price * 0.8)}</span>
            <span>{fmtUsd(rec.current_price * 1.2)}</span>
          </div>
        </div>
      </Surface>

      <div className="flex flex-col gap-6">
        <Surface variant={variant} tone="raised" className="flex flex-col gap-3 p-6">
          <div className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
            What you're testing
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
        </Surface>

        <Surface variant={variant} tone="raised" className="flex flex-col gap-4 p-6">
          <div className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
            Projected outcome
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Δ price" value={fmtPct(deltaPct)} positive={deltaPct >= 0} />
            <Stat label="Δ units" value={fmtPct(unitLift)} positive={unitLift >= 0} />
            <Stat label="Forecast units" value={fmtIntCompact(closest.units)} />
            <Stat
              label="Forecast revenue"
              value={fmtUsd(closest.revenue)}
              positive={closest.revenue >= baseRevenue}
            />
          </div>
        </Surface>

        <Surface variant={variant} tone="raised" className="flex flex-col gap-3 p-6">
          <div className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
            Constraint check
          </div>
          <ul className="flex flex-col gap-2">
            {flags.map((f) => (
              <motion.li
                key={f.label}
                animate={{ opacity: 1 }}
                className="flex items-center justify-between gap-2 text-sm"
              >
                <span>{f.label}</span>
                <span
                  className={
                    f.ok
                      ? "rounded-full bg-positive/15 px-2 py-0.5 text-xs font-medium text-positive"
                      : "rounded-full bg-negative/15 px-2 py-0.5 text-xs font-medium text-negative"
                  }
                >
                  {f.ok ? "ok" : "breach"}
                </span>
              </motion.li>
            ))}
          </ul>
        </Surface>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  positive,
}: {
  label: string;
  value: string;
  positive?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-md border border-border p-3">
      <span className="text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </span>
      <span
        className={
          positive === undefined
            ? "font-display text-xl font-semibold tabular"
            : positive
              ? "font-display text-xl font-semibold tabular text-positive"
              : "font-display text-xl font-semibold tabular text-negative"
        }
      >
        {value}
      </span>
    </div>
  );
}
