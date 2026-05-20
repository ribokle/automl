import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

import { THEMES } from "@/lib/theme/tokens";
import { Badge } from "@/components/ui/badge";

export default function HomePage() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-aurora-light dark:bg-aurora-dark opacity-70" />
      <div className="relative z-10 mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-16 md:py-24">
        <header className="flex flex-col gap-4">
          <Badge variant="outline" className="w-fit border-foreground/20 backdrop-blur">
            Client-facing prototypes
          </Badge>
          <h1 className="font-display text-4xl font-semibold tracking-tight md:text-6xl text-balance">
            Three directions for the AutoPrice client UI.
          </h1>
          <p className="max-w-2xl text-pretty text-lg text-muted-foreground">
            The same six surfaces — landing, dashboard, recommendations, simulate, validation,
            methodology — wrapped in three distinct visual languages. Pick one or mix.
          </p>
        </header>

        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {(["a", "b", "c"] as const).map((v) => {
            const t = THEMES[v];
            return (
              <Link
                key={v}
                href={`/proto/${v}/dashboard`}
                className="group relative flex flex-col gap-4 overflow-hidden rounded-2xl border border-foreground/10 bg-background/60 p-8 backdrop-blur transition-all hover:border-foreground/30 hover:shadow-raised"
              >
                <div className="flex items-baseline justify-between">
                  <span className="font-mono text-xs uppercase tracking-[0.2em] text-muted-foreground">
                    Variant {v.toUpperCase()}
                  </span>
                  <ArrowUpRight className="size-4 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:-translate-y-1" />
                </div>
                <div className="font-display text-2xl font-semibold tracking-tight md:text-3xl">
                  {t.name}
                </div>
                <p className="text-sm text-muted-foreground">{t.tagline}</p>
                <div className="mt-4 flex gap-2">
                  <span
                    className="size-6 rounded-full border border-foreground/10"
                    style={{ background: t.chartColors.primary }}
                  />
                  <span
                    className="size-6 rounded-full border border-foreground/10"
                    style={{ background: t.chartColors.secondary }}
                  />
                  <span
                    className="size-6 rounded-full border border-foreground/10"
                    style={{ background: t.chartColors.positive }}
                  />
                </div>
              </Link>
            );
          })}
        </div>

        <footer className="mt-auto pt-16 text-sm text-muted-foreground">
          Each variant ships light + dark mode. Toggle in the nav. Same content everywhere — the
          comparison is purely visual.
        </footer>
      </div>
    </main>
  );
}
