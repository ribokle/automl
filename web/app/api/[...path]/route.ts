// Catch-all proxy to the FastAPI backend.
//
// Replaces the rewrite that previously lived in next.config.mjs. Running as a
// Node route handler lets us inject `Authorization: Bearer <API_AUTH_TOKEN>`
// from server-side env without ever exposing the token to the client bundle.
// Streaming bodies (SSE in particular) are forwarded by passing the upstream
// response body straight through.

import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const UPSTREAM =
  process.env.API_PROXY_TARGET ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://localhost:8000";

const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailers",
  "transfer-encoding",
  "upgrade",
  "host",
  "content-length",
]);

function buildOutboundHeaders(req: NextRequest): Headers {
  const out = new Headers();
  req.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) out.set(key, value);
  });
  const token = process.env.API_AUTH_TOKEN;
  if (token) out.set("Authorization", `Bearer ${token}`);
  return out;
}

function buildInboundHeaders(res: Response): Headers {
  const out = new Headers();
  res.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) out.set(key, value);
  });
  return out;
}

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const search = req.nextUrl.search ?? "";
  const target = `${UPSTREAM}/${(path ?? []).join("/")}${search}`;

  const init: RequestInit & { duplex?: "half" } = {
    method: req.method,
    headers: buildOutboundHeaders(req),
    redirect: "manual",
  };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = req.body;
    init.duplex = "half";
  }

  const upstream = await fetch(target, init);
  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: buildInboundHeaders(upstream),
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const OPTIONS = proxy;
export const HEAD = proxy;
