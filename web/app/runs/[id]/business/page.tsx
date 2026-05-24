import { BusinessView } from "@/components/BusinessView";
import { RunSubnav } from "@/components/RunSubnav";

interface PageProps {
  params: { id: string };
}

export default function BusinessPage({ params }: PageProps) {
  return (
    <main className="space-y-4">
      <RunSubnav runId={params.id} />
      <BusinessView runId={params.id} />
    </main>
  );
}
