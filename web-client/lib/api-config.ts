const DEFAULT_UPSTREAM = "http://localhost:8000";

export const SERVER_UPSTREAM =
  process.env.API_PROXY_TARGET || process.env.NEXT_PUBLIC_API_BASE || DEFAULT_UPSTREAM;

export function getApiBase(): string {
  const isServer = typeof window === "undefined";
  return isServer ? SERVER_UPSTREAM : "/api";
}

export function getAuthHeaders(): Record<string, string> {
  if (typeof window !== "undefined") return {};
  const token = process.env.API_AUTH_TOKEN;
  return token ? { Authorization: `Bearer ${token}` } : {};
}
