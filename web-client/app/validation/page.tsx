import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";

import { loadClientPayload } from "@/lib/data";
import { AppNav } from "@/components/shell/AppNav";
import { AnchorNumber } from "@/components/shell/AnchorNumber";
import { Card } from "@/components/shell/Card";
import { ConfidenceForest } from "@/components/charts/ConfidenceForest";
import { cn } from "@/lib/cn";
import type { ValidationCheck } from "@/lib/types";

const STATUS_ICON: Record<ValidationCheck["status"], React.ReactNode> = {
  pass: <CheckCircle2 className="size-4 text-positive" />,
  warn: <AlertTriangle className="size-4 text-warning" />,
  fail: <XCircle className="size-4 text-negative" />,
};

const STATUS_BADGE: Record<ValidationCheck["status"], string> = {
  pass: "border-positive/30 bg-positive/10 text-positive",
  warn: "border-warning/30 bg-warning/10 text-warning",
  fail: "border-negative/30 bg-negative/10 text-negative",
};

export default async function ValidationPage() {
  const payload = await loadClientPayload();
  const passes = payload.validation.filter((v) => v.status === "pass").length;
  const total = payload.validation.length;

  return (
    <>
      <AppNav />
      <main className="mx-auto max-w-6xl px-6 pb-24 pt-10">
        <header className="flex flex-col gap-3 border-b border-hairline pb-10">
          <div className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
            Why we trust this
          </div>
          <h1 className="display text-3xl font-semibold tracking-tight md:text-4xl">
            The trust check
          </h1>
          <p className="max-w-2xl text-pretty text-muted-foreground">
            Every recommendation passes through seven gates before it lands on your dashboard —
            statistical, business-rule, and a literature check against the Hoch (1995) Dominick's
            ranges and the Bijmolt (2005) meta-analysis. Here's the receipt.
          </p>
        </header>

        <section className="py-10">
          <AnchorNumber
            label="Trust score"
            value={`${payload.trust_score}%`}
            delta={`${passes} / ${total} gates passed`}
            detail="Composite of statistical, business-rule, and reconciliation checks."
            size="xl"
          />
        </section>

        <section className="grid gap-3">
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
            Gates
          </div>
          <Card className="overflow-hidden">
            <ul className="divide-y divide-hairline">
              {payload.validation.map((c) => (
                <li key={c.name} className="grid grid-cols-12 items-center gap-4 px-6 py-4">
                  <div className="col-span-12 md:col-span-5 flex items-center gap-3">
                    {STATUS_ICON[c.status]}
                    <div>
                      <div className="font-medium">{c.name}</div>
                      <div className="text-xs text-muted-foreground">{c.note}</div>
                    </div>
                  </div>
                  <div className="col-span-8 md:col-span-5 text-sm text-muted-foreground"></div>
                  <div className="col-span-4 md:col-span-2 flex justify-end">
                    <span
                      className={cn(
                        "inline-flex rounded-full border px-2.5 py-0.5 text-xs font-medium tabular",
                        STATUS_BADGE[c.status],
                      )}
                    >
                      {c.metric}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </Card>
        </section>

        <section className="pt-12">
          <Card className="p-6">
            <div className="mb-4">
              <div className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
                Per-PPG elasticities
              </div>
              <h2 className="display mt-1 text-xl font-semibold">Confidence forest</h2>
              <p className="mt-1 max-w-xl text-sm text-muted-foreground">
                Point estimate per PPG with the model's uncertainty band. Anything to the right
                of −1 is inelastic — small price moves don't shift many units. The dashed line
                at −2.62 is the Bijmolt (2005) meta-analysis grand mean; bars outlined in warning
                colour sit outside the published category range.
              </p>
            </div>
            <ConfidenceForest data={payload.forest} />
          </Card>
        </section>
      </main>
    </>
  );
}
