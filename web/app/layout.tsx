import "./globals.css";
import type { Metadata, Viewport } from "next";
import Link from "next/link";

import { BackendStatus } from "@/components/BackendStatus";

export const metadata: Metadata = {
  title: "AutoML — Agentic Pricing",
  description: "Agentic price & promo optimization",
};

export const viewport: Viewport = {
  themeColor: "#0b0f1a",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur">
          <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-3">
            <Link href="/" className="flex items-baseline gap-2">
              <span className="text-sm font-bold tracking-tight text-slate-100">AutoML</span>
              <span className="text-[11px] text-slate-500">Agentic Pricing</span>
            </Link>
            <nav className="flex items-center gap-4 text-xs">
              <Link href="/" className="text-slate-300 hover:text-emerald-300">
                Home
              </Link>
              <Link href="/runs" className="text-slate-300 hover:text-emerald-300">
                Runs
              </Link>
              <BackendStatus />
            </nav>
          </div>
        </header>
        <div className="mx-auto max-w-6xl px-6 py-8">{children}</div>
      </body>
    </html>
  );
}
