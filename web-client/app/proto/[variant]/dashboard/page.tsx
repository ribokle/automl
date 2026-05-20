import { ArrowUpRight, Sparkles } from "lucide-react";
import Link from "next/link";

import { loadClientPayload } from "@/lib/data";
import { getTheme } from "@/lib/theme/tokens";
import { fmtUsdCompact, fmtPct } from "@/lib/format";
import { PageHeader } from "@/components/shell/PageHeader";
import { KpiTile } from "@/components/shell/KpiTile";
import { Surface } from "@/components/shell/Surface";
import { RevenueChart } from "@/components/charts/RevenueChart";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Recommendation, Variant } from "@/lib/types";

export default async function DashboardPage({ params }: { params: { variant: string } }) {
  const theme = getTheme(params.variant);
  const v = theme.variant;
  const payload = await loadClientPayload();

  const sortedByLift = [...payload.recommendations].sort(
    (a, b) => b.revenue_lift_usd - a.revenue_lift_usd,
  );
  const top = sortedByLift[0];
  const bottom = sortedByLift[sortedByLift.length - 1];

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        eyebrow={payload.run_id ? `Run · ${payload.run_id.slice(0, 8)}` : "Demo · synthetic data"}
        title="This week's optimised pricing"
        subtitle={payload.narrative}
        display={v === "c"}
      >
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <Badge variant="positive" className="gap-1.5">
            <Sparkles className="size-3" />
            Recommended for review
          </Badge>
          <Badge variant="outline">
            Generated {new Date(payload.generated_at).toLocaleString()}
          </Badge>
        </div>
      </PageHeader>

      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {payload.kpis.map((kpi, i) => (
          <KpiTile key={kpi.label} kpi={kpi} variant={v} index={i} />
        ))}
      </section>

      <section className="grid gap-6 lg:grid-cols-3">
        <Surface variant={v} tone="raised" className="lg:col-span-2 p-6">
          <div className="mb-4 flex items-baseline justify-between gap-4">
            <div>
              <div className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
                Forecast revenue
              </div>
              <h2 className="font-display text-xl font-semibold">
                Proposed vs status quo · 13 weeks
              </h2>
            </div>
            <Button variant="ghost" size="sm" asChild>
              <Link href={`/proto/${v}/recommendations`}>
                Open recommendations <ArrowUpRight className="size-3.5" />
              </Link>
            </Button>
          </div>
          <RevenueChart variant={v} data={payload.weekly} />
        </Surface>

        <Surface variant={v} tone="raised" className="flex flex-col gap-4 p-6">
          <div>
            <div className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
              Movers
            </div>
            <h2 className="font-display text-xl font-semibold">Biggest swings</h2>
          </div>
          <div className="flex flex-col gap-4">
            <MoverRow label="Top contributor" rec={top} v={v} />
            <MoverRow label="Smallest mover" rec={bottom} v={v} />
          </div>
          <div className="mt-auto pt-2 text-sm text-muted-foreground">
            Hover the chart for week-level detail. Open Recommendations for the per-PPG breakdown.
          </div>
        </Surface>
      </section>
    </div>
  );
}

function MoverRow({ label, rec, v: _v }: { label: string; rec: Recommendation; v: Variant }) {
  return (
    <div className="flex items-start gap-3 border-t border-border pt-4 first:border-t-0 first:pt-0">
      <div className="grid size-9 place-items-center rounded-md bg-muted font-mono text-xs">
        {rec.ppg_id.replace("ppg_", "")}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-3">
          <div className="truncate font-medium">{rec.ppg_name}</div>
          <div
            className={
              rec.delta_pct >= 0
                ? "tabular text-sm font-medium text-positive"
                : "tabular text-sm font-medium text-negative"
            }
          >
            {fmtPct(rec.delta_pct)}
          </div>
        </div>
        <div className="mt-0.5 flex items-baseline gap-2 text-xs text-muted-foreground">
          <span>{label}</span>
          <span>·</span>
          <span className="tabular">+{fmtUsdCompact(rec.revenue_lift_usd)} / Q</span>
        </div>
      </div>
    </div>
  );
}
