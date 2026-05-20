"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";

import { ColorOptionProvider } from "./ColorOptionProvider";
import { AppShell } from "./AppShell";
import type { ColorOption } from "@/lib/theme/tokens";

export function Providers({
  initialOption,
  initialMode,
  children,
}: {
  initialOption: ColorOption;
  initialMode: "light" | "dark";
  children: React.ReactNode;
}) {
  return (
    <NextThemesProvider attribute="class" defaultTheme={initialMode} enableSystem={false}>
      <ColorOptionProvider initial={initialOption}>
        <AppShell>{children}</AppShell>
      </ColorOptionProvider>
    </NextThemesProvider>
  );
}
