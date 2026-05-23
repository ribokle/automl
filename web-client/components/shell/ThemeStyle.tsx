import { COLOR_OPTIONS, type ColorOption } from "@/lib/theme/tokens";

function vars(record: Record<string, string>): string {
  return Object.entries(record)
    .map(([k, v]) => `${k}: ${v};`)
    .join("");
}

export function ThemeStyle({ option }: { option: ColorOption }) {
  const pack = COLOR_OPTIONS[option];
  const css = `
    [data-color="${option}"] {
      ${vars(pack.light)}
    }
    .dark [data-color="${option}"] {
      ${vars(pack.dark)}
    }
  `;
  return <style dangerouslySetInnerHTML={{ __html: css }} />;
}
