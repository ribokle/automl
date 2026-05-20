import { getApiBase, getAuthHeaders } from "./api-config";
import { buildMockPayload } from "./mock";
import type { ClientPayload, RunSummary } from "./types";

async function safeFetch<T>(url: string, init?: RequestInit): Promise<T | null> {
  try {
    const res = await fetch(url, {
      cache: "no-store",
      headers: { ...getAuthHeaders(), ...(init?.headers ?? {}) },
      ...init,
    });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export async function listRuns(): Promise<RunSummary[]> {
  const base = getApiBase();
  const runs = await safeFetch<RunSummary[]>(`${base}/runs?archived=false`);
  return runs ?? [];
}

export async function loadClientPayload(runId?: string | null): Promise<ClientPayload> {
  const runs = await listRuns();
  const target =
    (runId && runs.find((r) => r.id === runId)) ||
    runs.find((r) => r.status === "done") ||
    runs[0];

  if (!target) return buildMockPayload(null);

  return buildMockPayload(target.id);
}
