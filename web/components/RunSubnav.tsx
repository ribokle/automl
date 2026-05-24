"use client";

/**
 * Top navigation bar shared across run sub-pages
 * (/ → timeline, /eda → Advanced EDA, /business → Business view).
 *
 * Highlights the active tab by matching the current pathname.
 * Must be a client component so usePathname() is available.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";

interface Props {
  runId: string;
}

const TABS = [
  { label: "Timeline", href: (id: string) => `/runs/${id}` },
  { label: "Advanced EDA", href: (id: string) => `/runs/${id}/eda` },
  { label: "Business view", href: (id: string) => `/runs/${id}/business` },
] as const;

export function RunSubnav({ runId }: Props) {
  const pathname = usePathname();

  return (
    <nav className="flex flex-wrap items-center gap-1 border-b border-slate-800 pb-0">
      {TABS.map((tab) => {
        const href = tab.href(runId);
        const isActive = pathname === href;
        return (
          <Link
            key={href}
            href={href}
            className={`-mb-px border-b-2 px-4 py-2 text-xs font-medium transition-colors ${
              isActive
                ? "border-emerald-400 text-emerald-300"
                : "border-transparent text-slate-400 hover:border-slate-600 hover:text-slate-200"
            }`}
          >
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
