import { RunTimeline } from "@/components/RunTimeline";

interface PageProps {
  params: { id: string };
}

export default function RunPage({ params }: PageProps) {
  return (
    <main>
      <RunTimeline runId={params.id} />
    </main>
  );
}
