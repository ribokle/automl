import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react";

import { loadClientPayload } from "@/lib/data";
import { getTheme } from "@/lib/theme/tokens";
import { PageHeader } from "@/components/shell/PageHeader";
import { Surface } from "@/components/shell/Surface";
import { ConfidenceForest } from "@/components/charts/ConfidenceForest";
import { cn } from "@/lib/cn";
import type { ValidationCheck } from "@/lib/types";

const STATUS_ICON: Record<ValidationCheck["status"], React.ReactNode> = {
  pass: <CheckCircle2 className="size-4 text-positive" />,
  warn: <AlertTriangle className="size-4 text-warning" />,
  fail: <XCircle className="size-4 text-negative" />,
};

const STATUS_BADGE: Record<ValidationCheck["status"], string> = {
  pass: "bg-positive/15 text-positive",
  warn: "bg-warning/15 text-warning",
  fail: "bg-negative/15 text-negative",
};

export default async function ValidationPage({ params }: { params: { variant: string } }) {
  const theme = getTheme(params.variant);
  const v = theme.variant;
  const payload = await loadClientPayload();

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        eyebrow="Trust check"
        title="Why we trust this answer"
        subtitle="Before any recommendation lands on this page, it has to clear six gates. Here's the receipt."
        display={v === "c"}
      />

      <section className="grid gap-6 lg:grid-cols-2">
        <Surface variant={v} tone="raised" className="p-6">
          <div className="mb-4">
            <div className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
              Quality gates
            </div>
            <h2 className="font-display text-xl font-semibold">All checks</h2>
          </div>
          <ul className="flex flex-col gap-3">
            {payload.validation.map((c) => (
              <li
                key={c.name}
                className="flex items-start gap-3 rounded-md border border-border p-4"
              >
                <span className="mt-0.5">{STATUS_ICON[c.status]}</span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-3">
                    <div className="font-medium">{c.name}</div>
                    <span
                      className={cn(
                        "rounded-full px-2 py-0.5 text-xs font-medium tabular",
                        STATUS_BADGE[c.status],
                      )}
                    >
                      {c.metric}
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground">{c.note}</p>
                </div>
              </li>
            ))}
          </ul>
        </Surface>

        <Surface variant={v} tone="raised" className="p-6">
          <div className="mb-4">
            <div className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
              Per-PPG elasticities
            </div>
            <h2 className="font-display text-xl font-semibold">Confidence forest</h2>
            <p className="mt-1 max-w-md text-sm text-muted-foreground">
              Point estimate per PPG with the model's uncertainty band. The dashed line at -1 is the
              elastic / inelastic threshold.
            </p>
          </div>
          <ConfidenceForest variant={v} data={payload.forest} />
        </Surface>
      </section>
    </div>
  );
}
