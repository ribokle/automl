import { loadClientPayload } from "@/lib/data";
import { getTheme } from "@/lib/theme/tokens";
import { PageHeader } from "@/components/shell/PageHeader";
import { SimulateClient } from "./SimulateClient";

export default async function SimulatePage({ params }: { params: { variant: string } }) {
  const theme = getTheme(params.variant);
  const payload = await loadClientPayload();

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        eyebrow="What-if simulator"
        title="Push the price, see what changes"
        subtitle="Drag the slider to test any price on this Price-Pack Group. We replay the elasticity curve, recompute units and revenue, and re-check every guardrail."
        display={theme.variant === "c"}
      />
      <SimulateClient payload={payload} variant={theme.variant} />
    </div>
  );
}
