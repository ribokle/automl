import type { RunSummary } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function createRun(dataPath: string, gatesEnabled = false): Promise<RunSummary> {
  const res = await fetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data_path: dataPath, gates_enabled: gatesEnabled }),
  });
  if (!res.ok) throw new Error(`createRun failed: ${res.status}`);
  return res.json();
}

export async function listRuns(): Promise<RunSummary[]> {
  const res = await fetch(`${API_BASE}/runs`, { cache: "no-store" });
  if (!res.ok) throw new Error(`listRuns failed: ${res.status}`);
  return res.json();
}

export async function getRun(id: string): Promise<unknown> {
  const res = await fetch(`${API_BASE}/runs/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`getRun failed: ${res.status}`);
  return res.json();
}

export const eventsUrl = (id: string) => `${API_BASE}/runs/${id}/events`;
