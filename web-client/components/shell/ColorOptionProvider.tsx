"use client";

import * as React from "react";

import type { ColorOption } from "@/lib/theme/tokens";

interface ColorOptionContextValue {
  option: ColorOption;
  setOption: (next: ColorOption) => void;
}

const ColorOptionContext = React.createContext<ColorOptionContextValue | null>(null);

const COOKIE = "ap_color";

function setCookie(value: string) {
  if (typeof document === "undefined") return;
  document.cookie = `${COOKIE}=${value}; path=/; max-age=${60 * 60 * 24 * 365}; samesite=lax`;
}

export function ColorOptionProvider({
  initial,
  children,
}: {
  initial: ColorOption;
  children: React.ReactNode;
}) {
  const [option, setOptionState] = React.useState<ColorOption>(initial);

  const setOption = React.useCallback((next: ColorOption) => {
    setOptionState(next);
    setCookie(next);
  }, []);

  const value = React.useMemo(() => ({ option, setOption }), [option, setOption]);
  return <ColorOptionContext.Provider value={value}>{children}</ColorOptionContext.Provider>;
}

export function useColorOption() {
  const ctx = React.useContext(ColorOptionContext);
  if (!ctx) throw new Error("useColorOption must be used within ColorOptionProvider");
  return ctx;
}
