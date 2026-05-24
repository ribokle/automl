import Link from "next/link";
import { ArrowRight } from "lucide-react";

import { loadClientPayload } from "@/lib/data";
import { AppNav } from "@/components/shell/AppNav";
import { AnchorNumber } from "@/components/shell/AnchorNumber";
import { Button } from "@/components/ui/button";

export default async function HomePage({
  searchParams,
}: {
  searchParams: { runId?: string };
}) {
  const payload = await loadClientPayload(searchParams.runId);
  const repriced = payload.recommendations.filter((r) => Math.abs(r.delta_pct) > 0.005).length;

  return (
    <>
      <AppNav />
      <main className="mx-auto flex min-h-[calc(100vh-3.5rem)] max-w-6xl flex-col px-6">
        <div className="flex-1 grid items-center py-16 md:py-24">
          <div className="grid gap-10 md:grid-cols-[1.2fr_1fr]">
            <div className="flex flex-col gap-6">
              <div className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                Quarterly pricing review · 13-week horizon
              </div>
              <h1 className="display max-w-2xl text-balance text-4xl font-semibold tracking-tight md:text-5xl lg:text-6xl">
                We recommend repricing{" "}
                <span className="text-accent">{repriced} of {payload.recommendations.length}</span>{" "}
                of your Price-Pack Groups.
              </h1>
              <p className="max-w-xl text-pretty text-base text-muted-foreground md:text-lg">
                {payload.narrative}
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <Button asChild size="lg" className="gap-2">
                  <Link href="/dashboard">
                    Open the dashboard
                    <ArrowRight className="size-4" />
                  </Link>
                </Button>
                <Button asChild variant="ghost" size="lg">
                  <Link href="/methodology">How it works</Link>
                </Button>
              </div>
            </div>

            <div className="flex items-end">
              <AnchorNumber
                label={payload.anchor.label}
                value={payload.anchor.value}
                delta={payload.anchor.delta}
                detail={payload.anchor.detail}
                size="xl"
              />
            </div>
          </div>
        </div>
        <footer className="border-t border-hairline py-6 text-xs text-muted-foreground">
          {payload.run_id ? `Run ${payload.run_id.slice(0, 8)}` : "Synthetic demo data"} ·
          Generated {new Date(payload.generated_at).toLocaleString()}
        </footer>
      </main>
    </>
  );
}
