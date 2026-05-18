import type { PPGRow, PPGSelectionRow, RunStateFull, RunSummary } from "./types";

const isServer = typeof window === "undefined";
const SERVER_API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const API_BASE = isServer ? SERVER_API_BASE : "/api";

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

export async function getRun(id: string): Promise<RunStateFull> {
  const res = await fetch(`${API_BASE}/runs/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`getRun failed: ${res.status}`);
  return res.json();
}

export const eventsUrl = (id: string) => `${API_BASE}/runs/${id}/events`;

export const artifactUrl = (runId: string, name: string) =>
  `${API_BASE}/artifacts/${runId}/${name}`;

export async function approveAgent(runId: string, agent: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${runId}/approve?agent=${encodeURIComponent(agent)}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`approve failed: ${res.status}`);
}

export async function rejectAgent(runId: string, agent: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${runId}/reject?agent=${encodeURIComponent(agent)}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`reject failed: ${res.status}`);
}

export async function getPPGMappingTable(runId: string): Promise<PPGRow[] | null> {
  const res = await fetch(`${API_BASE}/artifacts/${runId}/ppg_mapping_table.json`, {
    cache: "no-store",
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`getPPGMappingTable failed: ${res.status}`);
  return res.json();
}

export async function getPPGSelection(runId: string): Promise<PPGSelectionRow[] | null> {
  const res = await fetch(`${API_BASE}/artifacts/${runId}/ppg_selection.json`, {
    cache: "no-store",
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`getPPGSelection failed: ${res.status}`);
  return res.json();
}

export async function getArtifact<T>(runId: string, name: string): Promise<T | null> {
  const res = await fetch(`${API_BASE}/artifacts/${runId}/${name}`, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`getArtifact(${name}) failed: ${res.status}`);
  return res.json();
}
