import { notFound } from "next/navigation";

import { THEMES, getTheme } from "@/lib/theme/tokens";
import { ThemeStyle } from "@/components/shell/ThemeStyle";
import { VariantNav } from "@/components/shell/VariantNav";
import { cn } from "@/lib/cn";

export function generateStaticParams() {
  return [{ variant: "a" }, { variant: "b" }, { variant: "c" }];
}

export default function VariantLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: { variant: string };
}) {
  if (!(params.variant in THEMES)) notFound();
  const theme = getTheme(params.variant);
  const v = theme.variant;

  const wrapperClass =
    v === "c"
      ? "relative min-h-screen bg-background"
      : v === "b"
        ? "relative min-h-screen bg-background"
        : "relative min-h-screen bg-background bg-grid";

  const navSurface =
    v === "c"
      ? "bg-background/40 backdrop-blur-xl"
      : v === "b"
        ? "bg-background/90 backdrop-blur"
        : "bg-background/85 backdrop-blur";

  return (
    <>
      <ThemeStyle variant={v} />
      <div data-variant={v} className={cn(wrapperClass, theme.bodyClass)}>
        {v === "c" ? (
          <div className="pointer-events-none fixed inset-0 -z-0 dark:bg-aurora-dark bg-aurora-light opacity-90" />
        ) : null}
        <div className="relative z-10">
          <VariantNav variant={v} variantName={theme.name} surfaceClass={navSurface} />
          <main className="mx-auto max-w-7xl px-6 pb-24">{children}</main>
        </div>
      </div>
    </>
  );
}
