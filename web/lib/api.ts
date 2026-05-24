import { getApiBase, getAuthHeaders } from "./api-config";
import type { PPGRow, PPGSelectionRow, RunStateFull, RunSummary } from "./types";

const API_BASE = getApiBase();
const authHeaders = getAuthHeaders;

export async function createRun(
  dataPath: string,
  gatesEnabled = false,
  agentMode = true,
  grainGateRequired = false,
): Promise<RunSummary> {
  const res = await fetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({
      data_path: dataPath,
      gates_enabled: gatesEnabled,
      agent_mode: agentMode,
      grain_gate_required: grainGateRequired,
    }),
  });
  if (!res.ok) throw new Error(`createRun failed: ${res.status}`);
  return res.json();
}

export interface UploadResult {
  path: string;
  bytes: number;
  filename: string;
}

export async function uploadCsv(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/uploads`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // body wasn't JSON; fall back to status
    }
    throw new Error(`upload failed: ${detail}`);
  }
  return res.json();
}

export async function getHealth(signal?: AbortSignal): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/health`, {
    cache: "no-store",
    headers: authHeaders(),
    signal,
  });
  if (!res.ok) throw new Error(`health failed: ${res.status}`);
  return res.json();
}

export async function listRuns(archived = false): Promise<RunSummary[]> {
  const url = `${API_BASE}/runs?archived=${archived ? "true" : "false"}`;
  const res = await fetch(url, { cache: "no-store", headers: authHeaders() });
  if (!res.ok) throw new Error(`listRuns failed: ${res.status}`);
  return res.json();
}

export async function archiveRun(id: string): Promise<RunSummary> {
  const res = await fetch(`${API_BASE}/runs/${id}/archive`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`archive failed: ${res.status}`);
  return res.json();
}

export async function unarchiveRun(id: string): Promise<RunSummary> {
  const res = await fetch(`${API_BASE}/runs/${id}/unarchive`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`unarchive failed: ${res.status}`);
  return res.json();
}

export async function deleteRun(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // not JSON
    }
    throw new Error(`delete failed: ${detail}`);
  }
}

export async function getRun(id: string): Promise<RunStateFull> {
  const res = await fetch(`${API_BASE}/runs/${id}`, { cache: "no-store", headers: authHeaders() });
  if (!res.ok) throw new Error(`getRun failed: ${res.status}`);
  return res.json();
}

export const eventsUrl = (id: string) => `${API_BASE}/runs/${id}/events`;

export const artifactUrl = (runId: string, name: string) =>
  `${API_BASE}/artifacts/${runId}/${name}`;

export interface ApprovePayload {
  modelling_grain?: string;
  comparison_grains?: string[];
}

export async function approveAgent(
  runId: string,
  agent: string,
  payload?: ApprovePayload,
): Promise<void> {
  const init: RequestInit = {
    method: "POST",
    headers: payload
      ? { "Content-Type": "application/json", ...authHeaders() }
      : authHeaders(),
  };
  if (payload) {
    init.body = JSON.stringify(payload);
  }
  const res = await fetch(
    `${API_BASE}/runs/${runId}/approve?agent=${encodeURIComponent(agent)}`,
    init,
  );
  if (!res.ok) throw new Error(`approve failed: ${res.status}`);
}

export async function rejectAgent(runId: string, agent: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${runId}/reject?agent=${encodeURIComponent(agent)}`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`reject failed: ${res.status}`);
}

export async function rerunAgent(
  runId: string,
  agent: string,
  options: Record<string, unknown>,
): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${runId}/rerun?agent=${encodeURIComponent(agent)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(options),
  });
  if (!res.ok) throw new Error(`rerun failed: ${res.status}`);
}

export async function getPPGMappingTable(runId: string): Promise<PPGRow[] | null> {
  const res = await fetch(`${API_BASE}/artifacts/${runId}/ppg_mapping_table.json`, {
    cache: "no-store",
    headers: authHeaders(),
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`getPPGMappingTable failed: ${res.status}`);
  return res.json();
}

export async function getPPGSelection(runId: string): Promise<PPGSelectionRow[] | null> {
  const res = await fetch(`${API_BASE}/artifacts/${runId}/ppg_selection.json`, {
    cache: "no-store",
    headers: authHeaders(),
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`getPPGSelection failed: ${res.status}`);
  return res.json();
}

export async function getQuerySchema(): Promise<unknown> {
  const res = await fetch(`${API_BASE}/runs/query/schema`, {
    cache: "force-cache",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`getQuerySchema failed: ${res.status}`);
  return res.json();
}

export async function runQuery<T = unknown>(runId: string, spec: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}/runs/${runId}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(spec),
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // not JSON
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function distinctValues(runId: string, column: string): Promise<(string | number)[]> {
  const res = await fetch(
    `${API_BASE}/runs/${runId}/query/distinct?column=${encodeURIComponent(column)}`,
    { cache: "no-store", headers: authHeaders() },
  );
  if (!res.ok) throw new Error(`distinctValues failed: ${res.status}`);
  return res.json();
}

export async function getArtifact<T>(runId: string, name: string): Promise<T | null> {
  const res = await fetch(`${API_BASE}/artifacts/${runId}/${name}`, {
    cache: "no-store",
    headers: authHeaders(),
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`getArtifact(${name}) failed: ${res.status}`);
  return res.json();
}
