import { RunSubnav } from "@/components/RunSubnav";
import { RunSidebar } from "@/components/RunSidebar";
import { RunTimeline } from "@/components/RunTimeline";

interface PageProps {
  params: { id: string };
}

export default function RunPage({ params }: PageProps) {
  return (
    <main className="flex flex-col gap-6 lg:flex-row lg:items-start">
      <RunSidebar activeRunId={params.id} />
      <div className="min-w-0 flex-1">
        <div className="mb-4">
          <RunSubnav runId={params.id} />
        </div>
        <RunTimeline runId={params.id} />
      </div>
    </main>
  );
}
