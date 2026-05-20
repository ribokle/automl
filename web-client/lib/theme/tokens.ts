export type ColorOption = "vault" | "marquee";
export type Mode = "light" | "dark";

export interface ColorPack {
  id: ColorOption;
  name: string;
  description: string;
  defaultMode: Mode;
  light: Record<string, string>;
  dark: Record<string, string>;
  chartColors: {
    primary: string;
    primaryAlt: string;
    secondary: string;
    positive: string;
    negative: string;
    grid: string;
  };
}

const SHARED_LIGHT: Record<string, string> = {
  "--radius": "0.5rem",
  "--font-sans": '"Inter", system-ui, sans-serif',
  "--font-display": '"Inter Tight", "Inter", system-ui, sans-serif',
  "--font-mono": '"JetBrains Mono", ui-monospace, monospace',
  "--shadow-card": "0 1px 2px rgb(0 0 0 / 0.04)",
  "--shadow-raised": "0 1px 2px rgb(0 0 0 / 0.04), 0 8px 24px rgb(0 0 0 / 0.04)",
  "--shadow-glow": "0 0 0 1px hsl(var(--accent) / 0.18)",
};

const SHARED_DARK: Record<string, string> = {
  "--shadow-card": "0 1px 2px rgb(0 0 0 / 0.5)",
  "--shadow-raised": "0 1px 2px rgb(0 0 0 / 0.5), 0 12px 40px rgb(0 0 0 / 0.4)",
  "--shadow-glow": "0 0 0 1px hsl(var(--accent) / 0.3)",
};

export const COLOR_OPTIONS: Record<ColorOption, ColorPack> = {
  vault: {
    id: "vault",
    name: "Vault",
    description: "Mercury-inspired. Cream + mint. Light-first.",
    defaultMode: "light",
    chartColors: {
      primary: "#4CC09C",
      primaryAlt: "#1F8F6E",
      secondary: "#94A3B8",
      positive: "#1F8F6E",
      negative: "#DC2626",
      grid: "rgba(15, 23, 42, 0.06)",
    },
    light: {
      ...SHARED_LIGHT,
      "--background": "48 33% 97%",
      "--surface": "0 0% 100%",
      "--raised": "0 0% 100%",
      "--foreground": "240 6% 5%",
      "--muted": "44 14% 94%",
      "--muted-foreground": "240 4% 38%",
      "--border": "44 8% 88%",
      "--hairline": "240 6% 5% / 0.06",
      "--ring": "160 47% 53%",
      "--accent": "160 47% 53%",
      "--accent-foreground": "160 60% 12%",
      "--positive": "160 64% 34%",
      "--negative": "0 72% 51%",
      "--warning": "35 92% 45%",
    },
    dark: {
      ...SHARED_DARK,
      "--background": "240 8% 4%",
      "--surface": "240 6% 8%",
      "--raised": "240 6% 10%",
      "--foreground": "0 0% 96%",
      "--muted": "240 5% 14%",
      "--muted-foreground": "240 5% 64%",
      "--border": "240 5% 16%",
      "--hairline": "0 0% 100% / 0.06",
      "--ring": "160 60% 60%",
      "--accent": "160 60% 60%",
      "--accent-foreground": "160 60% 8%",
      "--positive": "160 60% 55%",
      "--negative": "0 80% 65%",
      "--warning": "35 92% 60%",
    },
  },
  marquee: {
    id: "marquee",
    name: "Marquee",
    description: "Runway-inspired. Near-black + violet. Dark-first.",
    defaultMode: "dark",
    chartColors: {
      primary: "#A78BFA",
      primaryAlt: "#7C3AED",
      secondary: "#5EEAD4",
      positive: "#34D399",
      negative: "#FB7185",
      grid: "rgba(255,255,255,0.07)",
    },
    light: {
      ...SHARED_LIGHT,
      "--background": "240 20% 98%",
      "--surface": "0 0% 100%",
      "--raised": "0 0% 100%",
      "--foreground": "240 24% 6%",
      "--muted": "240 14% 95%",
      "--muted-foreground": "240 8% 38%",
      "--border": "240 14% 90%",
      "--hairline": "240 24% 6% / 0.06",
      "--ring": "262 83% 58%",
      "--accent": "262 83% 58%",
      "--accent-foreground": "0 0% 100%",
      "--positive": "160 64% 34%",
      "--negative": "350 80% 55%",
      "--warning": "35 92% 45%",
    },
    dark: {
      ...SHARED_DARK,
      "--background": "240 25% 5%",
      "--surface": "240 18% 8%",
      "--raised": "240 16% 10%",
      "--foreground": "0 0% 98%",
      "--muted": "240 12% 16%",
      "--muted-foreground": "240 6% 65%",
      "--border": "240 10% 18%",
      "--hairline": "0 0% 100% / 0.07",
      "--ring": "256 90% 76%",
      "--accent": "256 90% 76%",
      "--accent-foreground": "240 25% 5%",
      "--positive": "160 70% 55%",
      "--negative": "350 85% 70%",
      "--warning": "35 92% 65%",
    },
  },
};

export function getColorOption(value: string | undefined | null): ColorOption {
  return value === "marquee" ? "marquee" : "vault";
}
