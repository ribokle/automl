import { loadClientPayload } from "@/lib/data";
import { AppNav } from "@/components/shell/AppNav";
import { AnchorNumber } from "@/components/shell/AnchorNumber";
import { Card } from "@/components/shell/Card";
import { RecCard } from "@/components/shell/RecCard";
import { fmtUsdCompact } from "@/lib/format";

export default async function RecommendationsPage() {
  const payload = await loadClientPayload();
  const recs = [...payload.recommendations].sort(
    (a, b) => b.revenue_lift_usd - a.revenue_lift_usd,
  );
  const totalLift = recs.reduce((a, r) => a + r.revenue_lift_usd, 0);

  return (
    <>
      <AppNav />
      <main className="mx-auto max-w-6xl px-6 pb-24 pt-10">
        <header className="flex flex-col gap-3 border-b border-hairline pb-10">
          <div className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
            Per Price-Pack Group
          </div>
          <h1 className="display text-3xl font-semibold tracking-tight md:text-4xl">
            {recs.length} recommendations
          </h1>
          <p className="max-w-2xl text-pretty text-muted-foreground">
            Each row is one Price-Pack Group. Tap to see why we picked the price and what the
            model expects to happen.
          </p>
        </header>

        <section className="grid gap-10 py-10 md:grid-cols-[1fr_auto]">
          <AnchorNumber
            label="Combined quarterly revenue lift"
            value={`+${fmtUsdCompact(totalLift)}`}
            delta={`across ${recs.filter((r) => Math.abs(r.delta_pct) > 0.005).length} re-priced PPGs`}
          />
        </section>

        <Card className="overflow-hidden">
          <div className="grid grid-cols-12 gap-4 border-b border-hairline px-6 py-3 text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
            <div className="col-span-5">Price-Pack Group</div>
            <div className="col-span-3 text-right">Current → Proposed</div>
            <div className="col-span-2 text-right">Revenue / Q</div>
            <div className="col-span-2 text-right">Confidence</div>
          </div>
          <div>
            {recs.map((r) => (
              <RecCard key={r.ppg_id} rec={r} showExpander />
            ))}
          </div>
        </Card>
      </main>
    </>
  );
}
