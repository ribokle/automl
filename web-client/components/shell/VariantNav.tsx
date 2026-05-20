"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, BarChart3, Layers, ScanSearch, Sparkles } from "lucide-react";

import { cn } from "@/lib/cn";
import { ThemeToggle } from "./ThemeToggle";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const ROUTES = [
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { href: "/recommendations", label: "Recommendations", icon: Layers },
  { href: "/simulate", label: "Simulate", icon: Sparkles },
  { href: "/validation", label: "Validation", icon: ScanSearch },
  { href: "/methodology", label: "Methodology", icon: Activity },
];

const VARIANTS: Array<{ id: "a" | "b" | "c"; name: string }> = [
  { id: "a", name: "Premium-Dark" },
  { id: "b", name: "Executive-Bright" },
  { id: "c", name: "Spatial-Glass" },
];

interface Props {
  variant: "a" | "b" | "c";
  variantName: string;
  className?: string;
  surfaceClass?: string;
}

export function VariantNav({ variant, variantName, className, surfaceClass }: Props) {
  const pathname = usePathname();
  const base = `/proto/${variant}`;

  return (
    <nav
      className={cn(
        "sticky top-0 z-30 w-full border-b border-border/60",
        surfaceClass ?? "bg-background/85 backdrop-blur",
        className,
      )}
    >
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-4 px-6">
        <Link href={base} className="flex items-center gap-2 font-display font-semibold">
          <span className="grid size-7 place-items-center rounded-md bg-accent text-accent-foreground text-[11px] font-bold">
            AP
          </span>
          <span className="hidden sm:inline">AutoPrice</span>
        </Link>
        <span className="hidden text-xs text-muted-foreground sm:inline">/ {variantName}</span>
        <div className="ml-4 hidden items-center gap-1 lg:flex">
          {ROUTES.map((r) => {
            const href = `${base}${r.href}`;
            const active = pathname?.startsWith(href);
            const Icon = r.icon;
            return (
              <Link
                key={r.href}
                href={href}
                className={cn(
                  "inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-colors",
                  active
                    ? "bg-muted text-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                <Icon className="size-3.5" />
                {r.label}
              </Link>
            );
          })}
        </div>
        <div className="ml-auto flex items-center gap-1">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="text-xs">
                Variant: {variant.toUpperCase()}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="text-sm">
              {VARIANTS.map((v) => (
                <DropdownMenuItem key={v.id} asChild>
                  <Link href={`/proto/${v.id}${pathname?.split("/").slice(3).join("/") || "/dashboard"}`}>
                    {v.id.toUpperCase()} · {v.name}
                  </Link>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <ThemeToggle />
        </div>
      </div>
    </nav>
  );
}
