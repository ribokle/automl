import Link from "next/link";
import { ArrowRight } from "lucide-react";

import { loadClientPayload } from "@/lib/data";
import { AppNav } from "@/components/shell/AppNav";
import { AnchorNumber } from "@/components/shell/AnchorNumber";
import { Card, RaisedCard } from "@/components/shell/Card";
import { RecCard } from "@/components/shell/RecCard";
import { RevenueChart } from "@/components/charts/RevenueChart";
import { Button } from "@/components/ui/button";

export default async function DashboardPage({
  searchParams,
}: {
  searchParams: { runId?: string };
}) {
  const payload = await loadClientPayload(searchParams.runId);
  const top = [...payload.recommendations].sort(
    (a, b) => b.revenue_lift_usd - a.revenue_lift_usd,
  );

  return (
    <>
      <AppNav />
      <main className="mx-auto max-w-6xl px-6 pb-24 pt-10">
        <header className="flex flex-col gap-2 pb-10">
          <div className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
            Quarterly review · Week of {new Date(payload.generated_at).toLocaleDateString()}
          </div>
          <h1 className="display text-3xl font-semibold tracking-tight md:text-4xl">
            This quarter's pricing recommendation
          </h1>
          <p className="max-w-3xl text-pretty text-muted-foreground">{payload.narrative}</p>
        </header>

        <section className="grid gap-8 border-b border-hairline pb-10">
          <AnchorNumber
            label={payload.anchor.label}
            value={payload.anchor.value}
            delta={payload.anchor.delta}
            detail={payload.anchor.detail}
            size="xl"
          />
          <div className="grid gap-x-12 gap-y-6 sm:grid-cols-3">
            {payload.kpis.map((k) => (
              <div key={k.label} className="flex flex-col gap-1.5">
                <span className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
                  {k.label}
                </span>
                <span className="display text-2xl font-semibold tabular">{k.value}</span>
                <span className="text-xs text-muted-foreground">{k.hint}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="grid gap-6 pt-10 lg:grid-cols-5">
          <RaisedCard className="lg:col-span-3 p-6">
            <div className="mb-1 text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
              Forecast revenue · 13 weeks
            </div>
            <div className="mb-5 flex items-baseline gap-4">
              <span className="display text-2xl font-semibold">Proposed vs status quo</span>
            </div>
            <RevenueChart data={payload.weekly} />
          </RaisedCard>

          <Card className="lg:col-span-2 p-6">
            <div className="mb-5 flex items-baseline justify-between gap-3">
              <div>
                <div className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
                  Top recommendations
                </div>
                <h2 className="display mt-1 text-xl font-semibold">Biggest movers</h2>
              </div>
              <Button asChild variant="ghost" size="sm" className="gap-1">
                <Link href="/recommendations">
                  All <ArrowRight className="size-3.5" />
                </Link>
              </Button>
            </div>
            <div className="-mx-6">
              {top.slice(0, 3).map((rec) => (
                <RecCard key={rec.ppg_id} rec={rec} compact />
              ))}
            </div>
          </Card>
        </section>
      </main>
    </>
  );
}
