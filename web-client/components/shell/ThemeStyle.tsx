import { THEMES } from "@/lib/theme/tokens";
import type { Variant } from "@/lib/types";

function vars(record: Record<string, string>): string {
  return Object.entries(record)
    .map(([k, v]) => `${k}: ${v};`)
    .join("");
}

export function ThemeStyle({ variant }: { variant: Variant }) {
  const t = THEMES[variant];
  const css = `
    [data-variant="${variant}"] {
      ${vars(t.light)}
      --radius: ${t.radius};
      --font-sans: ${t.fonts.sans};
      --font-display: ${t.fonts.display};
      --font-mono: ${t.fonts.mono};
    }
    [data-variant="${variant}"].dark {
      ${vars(t.dark)}
    }
  `;
  return <style dangerouslySetInnerHTML={{ __html: css }} />;
}
