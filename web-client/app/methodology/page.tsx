import { loadClientPayload } from "@/lib/data";
import { AppNav } from "@/components/shell/AppNav";

export default async function MethodologyPage({
  searchParams,
}: {
  searchParams: { runId?: string };
}) {
  const payload = await loadClientPayload(searchParams.runId);

  return (
    <>
      <AppNav />
      <main className="mx-auto max-w-3xl px-6 pb-24 pt-10">
        <header className="flex flex-col gap-3 border-b border-hairline pb-10">
          <div className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
            How it works
          </div>
          <h1 className="display text-3xl font-semibold tracking-tight md:text-4xl">
            Seven steps, end to end
          </h1>
          <p className="max-w-2xl text-pretty text-muted-foreground">
            No black box. Every recommendation passes through a numbered sequence — here's the
            short version.
          </p>
        </header>

        <ol className="flex flex-col">
          {payload.methodology.map((step, i) => (
            <li
              key={step.agent}
              className="grid grid-cols-[auto_1fr] items-start gap-x-6 gap-y-2 border-b border-hairline py-8 last:border-b-0"
            >
              <div className="flex h-full flex-col items-center">
                <div className="font-mono text-xs font-medium tabular text-muted-foreground">
                  {String(i + 1).padStart(2, "0")}
                </div>
              </div>
              <div className="flex flex-col gap-2">
                <div className="text-[11px] font-medium uppercase tracking-[0.16em] text-muted-foreground">
                  {step.agent}
                </div>
                <h3 className="display text-xl font-semibold">{step.title}</h3>
                <p className="text-pretty text-muted-foreground">{step.description}</p>
                <p className="text-sm font-medium text-accent">{step.implication}</p>
              </div>
            </li>
          ))}
        </ol>
      </main>
    </>
  );
}
