"use client";

import * as React from "react";
import { animate, useMotionValue, useTransform } from "framer-motion";

export function useAnimatedNumber(
  target: number,
  format: (n: number) => string,
  duration = 0.28,
): string {
  const mv = useMotionValue(target);
  const text = useTransform(mv, (v) => format(v));
  const [snapshot, setSnapshot] = React.useState<string>(format(target));

  React.useEffect(() => {
    const controls = animate(mv, target, { duration, ease: "easeOut" });
    return () => controls.stop();
  }, [target, duration, mv]);

  React.useEffect(() => {
    return text.on("change", (v) => setSnapshot(v));
  }, [text]);

  return snapshot;
}
