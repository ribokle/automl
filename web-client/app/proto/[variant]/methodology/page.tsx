import { loadClientPayload } from "@/lib/data";
import { getTheme } from "@/lib/theme/tokens";
import { PageHeader } from "@/components/shell/PageHeader";
import { Surface } from "@/components/shell/Surface";

export default async function MethodologyPage({ params }: { params: { variant: string } }) {
  const theme = getTheme(params.variant);
  const v = theme.variant;
  const payload = await loadClientPayload();

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        eyebrow="How it works"
        title="Seven steps, end to end"
        subtitle="No black box. Every recommendation passes through a numbered sequence — here's the short version."
        display={v === "c"}
      />

      <Surface variant={v} tone="raised" className="p-8">
        <ol className="flex flex-col gap-6">
          {payload.methodology.map((step, i) => (
            <li key={step.agent} className="grid grid-cols-[auto_1fr] items-start gap-6">
              <div className="flex flex-col items-center">
                <div className="grid size-9 place-items-center rounded-full bg-accent text-accent-foreground font-mono text-xs font-semibold">
                  {String(i + 1).padStart(2, "0")}
                </div>
                {i < payload.methodology.length - 1 ? (
                  <div className="mt-2 h-16 w-px bg-border" />
                ) : null}
              </div>
              <div className="flex flex-col gap-1 pt-1">
                <div className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
                  {step.agent}
                </div>
                <h3 className="font-display text-lg font-semibold">{step.title}</h3>
                <p className="max-w-2xl text-pretty text-sm text-muted-foreground">
                  {step.description}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </Surface>
    </div>
  );
}
