"use client";

import * as React from "react";
import { ArrowDown, ArrowRight, ArrowUp, ChevronDown, AlertTriangle } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";

import { cn } from "@/lib/cn";
import { fmtUsd, fmtUsdCompact, fmtPct, fmtPctRaw } from "@/lib/format";
import { Button } from "@/components/ui/button";
import type { Recommendation } from "@/lib/types";

interface Props {
  rec: Recommendation;
  compact?: boolean;
  showExpander?: boolean;
}

export function RecCard({ rec, compact = false, showExpander = false }: Props) {
  const [open, setOpen] = React.useState(false);
  const up = rec.delta_pct >= 0;

  return (
    <div
      className={cn(
        "group relative grid grid-cols-12 items-center gap-4 border-t border-hairline px-6 transition-colors first:border-t-0 hover:bg-muted/40",
        compact ? "py-3" : "py-4",
      )}
    >
      <div className="col-span-12 md:col-span-5 min-w-0">
        <div className="flex items-center gap-3">
          <div className="grid size-9 shrink-0 place-items-center rounded-md bg-muted font-mono text-xs">
            {rec.ppg_id.replace("ppg_", "")}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate font-medium">{rec.ppg_name}</span>
              {rec.flagged ? (
                <AlertTriangle className="size-3.5 shrink-0 text-warning" />
              ) : null}
            </div>
            <div className="text-xs text-muted-foreground">{rec.category}</div>
          </div>
        </div>
        {!compact ? (
          <p className="mt-2 line-clamp-1 text-xs text-muted-foreground">{rec.rationale}</p>
        ) : null}
      </div>

      <div className="col-span-7 md:col-span-3 flex items-center justify-end gap-2 text-sm tabular">
        <span className="text-muted-foreground">{fmtUsd(rec.current_price)}</span>
        <ArrowRight className="size-3 text-muted-foreground" />
        <span className="font-semibold">{fmtUsd(rec.proposed_price)}</span>
        <span
          className={cn(
            "inline-flex items-center gap-0.5 rounded-full border px-1.5 py-0.5 text-[11px] font-medium",
            up
              ? "border-positive/30 bg-positive/10 text-positive"
              : "border-negative/30 bg-negative/10 text-negative",
          )}
        >
          {up ? <ArrowUp className="size-2.5" /> : <ArrowDown className="size-2.5" />}
          {fmtPct(rec.delta_pct).replace("+", "")}
        </span>
      </div>

      <div className="col-span-5 md:col-span-2 text-right text-sm font-semibold tabular text-positive">
        +{fmtUsdCompact(rec.revenue_lift_usd)}
      </div>

      <div className="col-span-7 md:col-span-2 flex items-center justify-end gap-2">
        <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-accent"
            style={{ width: `${rec.confidence * 100}%` }}
          />
        </div>
        <span className="w-9 text-right text-[11px] tabular text-muted-foreground">
          {fmtPctRaw(rec.confidence, 0)}
        </span>
        {showExpander ? (
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setOpen((v) => !v)}
            aria-label={open ? "Collapse" : "Expand"}
            className="size-7"
          >
            <ChevronDown
              className={cn("size-4 transition-transform", open ? "rotate-180" : "")}
            />
          </Button>
        ) : null}
      </div>

      {showExpander ? (
        <AnimatePresence initial={false}>
          {open ? (
            <motion.div
              key="detail"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="col-span-12 overflow-hidden"
            >
              <div className="grid grid-cols-2 gap-x-8 gap-y-3 border-t border-hairline pt-4 text-sm md:grid-cols-4">
                <Stat label="Elasticity" value={rec.elasticity.toFixed(2)} />
                <Stat label="Δ units" value={fmtPct(rec.unit_lift_pct)} />
                <Stat
                  label="Margin (proposed)"
                  value={`${((rec.proposed_price - rec.current_price * 0.6) / rec.proposed_price * 100).toFixed(1)}%`}
                />
                <Stat label="Confidence" value={fmtPctRaw(rec.confidence, 0)} />
                <div className="col-span-2 md:col-span-4 text-sm text-muted-foreground">
                  {rec.rationale}
                </div>
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </span>
      <span className="font-medium tabular">{value}</span>
    </div>
  );
}
