import * as React from "react";

import { cn } from "@/lib/cn";
import type { Variant } from "@/lib/types";

interface Props extends React.HTMLAttributes<HTMLDivElement> {
  variant: Variant;
  tone?: "default" | "raised";
}

export function Surface({ variant, tone = "default", className, children, ...rest }: Props) {
  const base = "rounded-lg border";
  let look = "";
  if (variant === "a") {
    look =
      tone === "raised"
        ? "bg-raised border-border shadow-card"
        : "bg-surface border-border shadow-card";
  } else if (variant === "b") {
    look =
      tone === "raised"
        ? "bg-surface border-border shadow-raised rounded-2xl"
        : "bg-surface border-border shadow-card rounded-2xl";
  } else {
    look =
      "dark:glass-dark glass-light rounded-2xl border-transparent shadow-raised";
  }
  return (
    <div className={cn(base, look, className)} {...rest}>
      {children}
    </div>
  );
}
