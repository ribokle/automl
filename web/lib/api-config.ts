// Centralised API base + auth-header resolution.
//
// Browser code always hits the same-origin /api path; the Node route handler
// at app/api/[...path]/route.ts proxies to the FastAPI upstream. The upstream
// URL is resolved server-side only so the client bundle stays host-agnostic.

const DEFAULT_UPSTREAM = "http://localhost:8000";

export const SERVER_UPSTREAM =
  process.env.API_PROXY_TARGET || process.env.NEXT_PUBLIC_API_BASE || DEFAULT_UPSTREAM;

export function getApiBase(): string {
  const isServer = typeof window === "undefined";
  return isServer ? SERVER_UPSTREAM : "/api";
}

export function getAuthHeaders(): Record<string, string> {
  // Only the Node runtime should see API_AUTH_TOKEN. In the browser bundle
  // this env var is undefined (and must stay that way — never NEXT_PUBLIC_*).
  if (typeof window !== "undefined") return {};
  const token = process.env.API_AUTH_TOKEN;
  return token ? { Authorization: `Bearer ${token}` } : {};
}
