"use client";

/**
 * Per-card grain selection state.
 *
 * Reads `run.options.{modelling_grain, comparison_grains}` once the run is
 * ready, then exposes a selection state + a `nameFor(base)` helper that
 * tacks `__<grain>` onto canonical artifact filenames when the operator
 * has toggled a comparison grain. Pre-grain cards don't use this — they
 * read canonical names directly.
 *
 * Defaults are safe: when no comparison grains are configured, every
 * `nameFor(...)` returns the canonical name, so cards that adopt this
 * hook show no visual diff for non-multi-grain runs.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { getRun } from "./api";

export interface GrainState {
  primary: string;
  comparisons: string[];
  selected: string;
  isPrimary: boolean;
  setSelected: (g: string) => void;
  nameFor: (base: string) => string;
}

export function useGrainState(runId: string, ready: boolean): GrainState {
  const [primary, setPrimary] = useState<string>("ppg_week");
  const [comparisons, setComparisons] = useState<string[]>([]);
  const [selected, setSelected] = useState<string>("ppg_week");

  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    getRun(runId)
      .then((s) => {
        if (cancelled) return;
        const opts = (s.options ?? {}) as Record<string, unknown>;
        const p =
          typeof opts.modelling_grain === "string" ? opts.modelling_grain : "ppg_week";
        const c = Array.isArray(opts.comparison_grains)
          ? (opts.comparison_grains as string[]).filter((g) => g !== p)
          : [];
        setPrimary(p);
        setComparisons(c);
        setSelected((cur) => (cur === p || c.includes(cur) ? cur : p));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [runId, ready]);

  const isPrimary = selected === primary;

  const nameFor = useCallback(
    (base: string) => {
      if (isPrimary) return base;
      const dot = base.lastIndexOf(".");
      if (dot <= 0) return `${base}__${selected}`;
      return `${base.slice(0, dot)}__${selected}${base.slice(dot)}`;
    },
    [isPrimary, selected],
  );

  return useMemo(
    () => ({ primary, comparisons, selected, isPrimary, setSelected, nameFor }),
    [primary, comparisons, selected, isPrimary, nameFor],
  );
}
