"use client";

import * as React from "react";

import { COLOR_OPTIONS } from "@/lib/theme/tokens";
import { ThemeStyle } from "./ThemeStyle";
import { useColorOption } from "./ColorOptionProvider";

export function AppShell({ children }: { children: React.ReactNode }) {
  const { option } = useColorOption();
  return (
    <>
      <ThemeStyle option={option} />
      <div data-color={option} className="min-h-screen bg-background text-foreground">
        {children}
      </div>
    </>
  );
}

export function ColorStyleAll() {
  return (
    <>
      {(Object.keys(COLOR_OPTIONS) as Array<keyof typeof COLOR_OPTIONS>).map((id) => (
        <ThemeStyle key={id} option={id} />
      ))}
    </>
  );
}
