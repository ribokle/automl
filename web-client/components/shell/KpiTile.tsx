"use client";

import { motion } from "framer-motion";

import { cn } from "@/lib/cn";
import { Surface } from "./Surface";
import type { Kpi, Variant } from "@/lib/types";

interface Props {
  kpi: Kpi;
  variant: Variant;
  index: number;
}

export function KpiTile({ kpi, variant, index }: Props) {
  const valueClass =
    variant === "c"
      ? "font-display text-5xl md:text-6xl font-semibold tracking-tight tabular"
      : variant === "b"
        ? "font-display text-4xl font-semibold tracking-tight tabular"
        : "font-display text-3xl font-semibold tracking-tight tabular";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, delay: index * 0.05, ease: "easeOut" }}
    >
      <Surface variant={variant} tone="raised" className="flex h-full flex-col gap-3 p-6">
        <div className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
          {kpi.label}
        </div>
        <div className="flex items-baseline gap-3">
          <span className={valueClass}>{kpi.value}</span>
          {kpi.delta ? (
            <span
              className={cn(
                "text-sm font-medium tabular",
                kpi.positive ? "text-positive" : "text-muted-foreground",
              )}
            >
              {kpi.delta}
            </span>
          ) : null}
        </div>
        {kpi.hint ? (
          <p className="text-pretty text-sm text-muted-foreground">{kpi.hint}</p>
        ) : null}
      </Surface>
    </motion.div>
  );
}
