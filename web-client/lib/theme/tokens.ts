import type { Variant } from "@/lib/types";

export interface ThemePack {
  variant: Variant;
  name: string;
  tagline: string;
  fonts: {
    sans: string;
    display: string;
    mono: string;
  };
  light: Record<string, string>;
  dark: Record<string, string>;
  radius: string;
  shellClass: string;
  bodyClass: string;
  chartColors: {
    primary: string;
    secondary: string;
    positive: string;
    negative: string;
    grid: string;
  };
}

const INTER = '"Inter", "Inter Tight", system-ui, sans-serif';
const INTER_TIGHT = '"Inter Tight", "Inter", system-ui, sans-serif';
const GEIST_FALLBACK = '"Inter", system-ui, sans-serif';
const JBM = '"JetBrains Mono", ui-monospace, "Cascadia Code", monospace';

export const THEMES: Record<Variant, ThemePack> = {
  a: {
    variant: "a",
    name: "Premium-Dark",
    tagline: "Linear / Vercel. Mono-accent, geometric, dense.",
    fonts: { sans: GEIST_FALLBACK, display: GEIST_FALLBACK, mono: JBM },
    radius: "0.5rem",
    shellClass: "",
    bodyClass: "selection:bg-emerald-400/25",
    chartColors: {
      primary: "#10F2A5",
      secondary: "#A78BFA",
      positive: "#10F2A5",
      negative: "#FF5C7C",
      grid: "rgba(255,255,255,0.06)",
    },
    light: {
      "--background": "0 0% 98%",
      "--surface": "0 0% 100%",
      "--raised": "0 0% 100%",
      "--foreground": "240 6% 10%",
      "--muted": "240 5% 96%",
      "--muted-foreground": "240 4% 36%",
      "--border": "240 6% 90%",
      "--ring": "158 80% 38%",
      "--accent": "158 80% 38%",
      "--accent-foreground": "0 0% 100%",
      "--positive": "158 80% 38%",
      "--negative": "350 80% 56%",
      "--warning": "38 92% 50%",
    },
    dark: {
      "--background": "240 8% 4%",
      "--surface": "240 6% 7%",
      "--raised": "240 6% 9%",
      "--foreground": "0 0% 96%",
      "--muted": "240 5% 14%",
      "--muted-foreground": "240 5% 64%",
      "--border": "240 5% 16%",
      "--ring": "158 89% 50%",
      "--accent": "158 89% 50%",
      "--accent-foreground": "240 8% 4%",
      "--positive": "158 89% 50%",
      "--negative": "350 89% 60%",
      "--warning": "38 92% 60%",
    },
  },
  b: {
    variant: "b",
    name: "Executive-Bright",
    tagline: "Stripe / Notion. Airy, indigo, projector-ready.",
    fonts: { sans: INTER, display: INTER_TIGHT, mono: JBM },
    radius: "1rem",
    shellClass: "",
    bodyClass: "",
    chartColors: {
      primary: "#4F46E5",
      secondary: "#D97706",
      positive: "#059669",
      negative: "#DC2626",
      grid: "rgba(15, 23, 42, 0.08)",
    },
    light: {
      "--background": "40 33% 98%",
      "--surface": "0 0% 100%",
      "--raised": "0 0% 100%",
      "--foreground": "24 10% 10%",
      "--muted": "30 10% 96%",
      "--muted-foreground": "24 7% 35%",
      "--border": "30 8% 90%",
      "--ring": "239 84% 60%",
      "--accent": "239 84% 60%",
      "--accent-foreground": "0 0% 100%",
      "--positive": "160 84% 30%",
      "--negative": "0 72% 51%",
      "--warning": "35 92% 45%",
    },
    dark: {
      "--background": "240 10% 7%",
      "--surface": "240 8% 11%",
      "--raised": "240 8% 13%",
      "--foreground": "0 0% 96%",
      "--muted": "240 6% 18%",
      "--muted-foreground": "240 5% 70%",
      "--border": "240 6% 22%",
      "--ring": "234 89% 74%",
      "--accent": "234 89% 74%",
      "--accent-foreground": "240 10% 7%",
      "--positive": "160 70% 55%",
      "--negative": "0 80% 65%",
      "--warning": "35 92% 60%",
    },
  },
  c: {
    variant: "c",
    name: "Spatial-Glass",
    tagline: "visionOS / Arc. Frosted glass, aurora, big type.",
    fonts: { sans: INTER, display: INTER_TIGHT, mono: JBM },
    radius: "1.25rem",
    shellClass: "",
    bodyClass: "",
    chartColors: {
      primary: "#34D399",
      secondary: "#A78BFA",
      positive: "#34D399",
      negative: "#FB7185",
      grid: "rgba(255,255,255,0.08)",
    },
    light: {
      "--background": "260 50% 98%",
      "--surface": "0 0% 100%",
      "--raised": "0 0% 100%",
      "--foreground": "222 47% 11%",
      "--muted": "260 30% 95%",
      "--muted-foreground": "222 14% 40%",
      "--border": "260 20% 88%",
      "--ring": "162 73% 46%",
      "--accent": "162 73% 46%",
      "--accent-foreground": "0 0% 100%",
      "--positive": "162 73% 46%",
      "--negative": "350 89% 60%",
      "--warning": "38 92% 50%",
    },
    dark: {
      "--background": "230 50% 6%",
      "--surface": "230 30% 10%",
      "--raised": "230 25% 12%",
      "--foreground": "210 40% 98%",
      "--muted": "230 20% 18%",
      "--muted-foreground": "215 20% 70%",
      "--border": "230 15% 22%",
      "--ring": "162 84% 55%",
      "--accent": "162 84% 55%",
      "--accent-foreground": "230 50% 6%",
      "--positive": "162 84% 55%",
      "--negative": "350 89% 65%",
      "--warning": "38 92% 65%",
    },
  },
};

export function getTheme(variant: string | undefined): ThemePack {
  if (variant === "b" || variant === "c") return THEMES[variant];
  return THEMES.a;
}
