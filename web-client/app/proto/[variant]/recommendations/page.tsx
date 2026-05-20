import { ArrowDown, ArrowUp, AlertTriangle } from "lucide-react";

import { loadClientPayload } from "@/lib/data";
import { getTheme } from "@/lib/theme/tokens";
import { fmtUsd, fmtUsdCompact, fmtPct, fmtPctRaw } from "@/lib/format";
import { PageHeader } from "@/components/shell/PageHeader";
import { Surface } from "@/components/shell/Surface";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cn";

export default async function RecommendationsPage({
  params,
}: {
  params: { variant: string };
}) {
  const theme = getTheme(params.variant);
  const v = theme.variant;
  const payload = await loadClientPayload();
  const recs = [...payload.recommendations].sort(
    (a, b) => b.revenue_lift_usd - a.revenue_lift_usd,
  );

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        eyebrow="Per Price-Pack Group"
        title="Recommended prices"
        subtitle="Each row is one Price-Pack Group. We show what we propose, what we expect it to do, and the model's confidence in that call."
        display={v === "c"}
      />

      <Surface variant={v} tone="raised" className="overflow-hidden">
        <div className="grid grid-cols-12 gap-4 border-b border-border px-6 py-3 text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
          <div className="col-span-4">PPG</div>
          <div className="col-span-2 text-right">Current → Proposed</div>
          <div className="col-span-1 text-right">Δ price</div>
          <div className="col-span-1 text-right">Δ units</div>
          <div className="col-span-2 text-right">Revenue / Q</div>
          <div className="col-span-2 text-right">Confidence</div>
        </div>
        <ul className="divide-y divide-border">
          {recs.map((r) => (
            <li
              key={r.ppg_id}
              className="grid grid-cols-12 items-center gap-4 px-6 py-4 transition-colors hover:bg-muted/40"
            >
              <div className="col-span-4 min-w-0">
                <div className="flex items-center gap-3">
                  <div className="grid size-9 shrink-0 place-items-center rounded-md bg-muted font-mono text-xs">
                    {r.ppg_id.replace("ppg_", "")}
                  </div>
                  <div className="min-w-0">
                    <div className="truncate font-medium">{r.ppg_name}</div>
                    <div className="text-xs text-muted-foreground">{r.category}</div>
                  </div>
                  {r.flagged ? (
                    <Badge variant="warning" className="gap-1">
                      <AlertTriangle className="size-3" />
                      review
                    </Badge>
                  ) : null}
                </div>
                <p className="mt-2 line-clamp-1 text-xs text-muted-foreground">{r.rationale}</p>
              </div>
              <div className="col-span-2 text-right text-sm tabular">
                <div className="text-muted-foreground line-through">{fmtUsd(r.current_price)}</div>
                <div className="font-semibold">{fmtUsd(r.proposed_price)}</div>
              </div>
              <div
                className={cn(
                  "col-span-1 text-right text-sm font-medium tabular",
                  r.delta_pct >= 0 ? "text-positive" : "text-negative",
                )}
              >
                <span className="inline-flex items-center gap-1">
                  {r.delta_pct >= 0 ? (
                    <ArrowUp className="size-3" />
                  ) : (
                    <ArrowDown className="size-3" />
                  )}
                  {fmtPct(r.delta_pct, 1).replace("+", "")}
                </span>
              </div>
              <div
                className={cn(
                  "col-span-1 text-right text-sm font-medium tabular",
                  r.unit_lift_pct >= 0 ? "text-positive" : "text-negative",
                )}
              >
                {fmtPct(r.unit_lift_pct)}
              </div>
              <div className="col-span-2 text-right text-sm font-semibold tabular text-positive">
                +{fmtUsdCompact(r.revenue_lift_usd)}
              </div>
              <div className="col-span-2 flex items-center justify-end gap-2">
                <div className="h-2 w-20 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-accent"
                    style={{ width: `${r.confidence * 100}%` }}
                  />
                </div>
                <span className="w-10 text-right text-xs tabular text-muted-foreground">
                  {fmtPctRaw(r.confidence, 0)}
                </span>
              </div>
            </li>
          ))}
        </ul>
      </Surface>
    </div>
  );
}
