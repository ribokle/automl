import { NextRequest } from "next/server";

import { SERVER_UPSTREAM } from "@/lib/api-config";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

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

function outbound(req: NextRequest): Headers {
  const out = new Headers();
  req.headers.forEach((v, k) => {
    if (!HOP_BY_HOP.has(k.toLowerCase())) out.set(k, v);
  });
  const token = process.env.API_AUTH_TOKEN;
  if (token) out.set("Authorization", `Bearer ${token}`);
  return out;
}

function inbound(res: Response): Headers {
  const out = new Headers();
  res.headers.forEach((v, k) => {
    if (!HOP_BY_HOP.has(k.toLowerCase())) out.set(k, v);
  });
  return out;
}

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const search = req.nextUrl.search ?? "";
  const target = `${SERVER_UPSTREAM}/${(path ?? []).join("/")}${search}`;
  const init: RequestInit & { duplex?: "half" } = {
    method: req.method,
    headers: outbound(req),
    redirect: "manual",
  };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = req.body;
    init.duplex = "half";
  }
  try {
    const up = await fetch(target, init);
    return new Response(up.body, {
      status: up.status,
      statusText: up.statusText,
      headers: inbound(up),
    });
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "upstream_unreachable", target, message: String(err) }),
      { status: 502, headers: { "content-type": "application/json" } },
    );
  }
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const OPTIONS = proxy;
export const HEAD = proxy;
