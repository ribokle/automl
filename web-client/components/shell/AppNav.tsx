"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/cn";
import { ThemeToggle } from "./ThemeToggle";
import { ColorToggle } from "./ColorToggle";

const ROUTES = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/recommendations", label: "Recommendations" },
  { href: "/simulate", label: "Simulate" },
  { href: "/validation", label: "Validation" },
  { href: "/methodology", label: "How it works" },
];

export function AppNav() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-30 w-full border-b border-hairline bg-background/85 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-6">
        <Link href="/" className="flex items-center gap-2 font-display text-[15px] font-semibold tracking-tight">
          <span className="grid size-7 place-items-center rounded-md bg-accent text-accent-foreground text-[11px] font-bold">
            AP
          </span>
          AutoPrice
        </Link>
        <div className="hidden items-center gap-0.5 md:flex">
          {ROUTES.map((r) => {
            const active = pathname === r.href || pathname?.startsWith(r.href + "/");
            return (
              <Link
                key={r.href}
                href={r.href}
                className={cn(
                  "rounded-md px-3 py-1.5 text-sm transition-colors",
                  active
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {r.label}
              </Link>
            );
          })}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <ColorToggle />
          <ThemeToggle />
        </div>
      </div>
    </nav>
  );
}
