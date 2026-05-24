import { loadClientPayload } from "@/lib/data";
import { AppNav } from "@/components/shell/AppNav";
import { SimulateClient } from "./SimulateClient";

export default async function SimulatePage({
  searchParams,
}: {
  searchParams: { runId?: string };
}) {
  const payload = await loadClientPayload(searchParams.runId);

  return (
    <>
      <AppNav />
      <main className="mx-auto max-w-6xl px-6 pb-24 pt-10">
        <header className="flex flex-col gap-3 border-b border-hairline pb-10">
          <div className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
            What-if simulator
          </div>
          <h1 className="display text-3xl font-semibold tracking-tight md:text-4xl">
            Push the price, watch the numbers move
          </h1>
          <p className="max-w-2xl text-pretty text-muted-foreground">
            Drag the sliders to test any price on this Price-Pack Group. Units, revenue, and the
            guardrails recompute in real time.
          </p>
        </header>
        <section className="pt-10">
          <SimulateClient payload={payload} />
        </section>
      </main>
    </>
  );
}
