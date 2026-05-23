"use client";

import { cn } from "@/lib/cn";
import { COLOR_OPTIONS } from "@/lib/theme/tokens";
import { useColorOption } from "./ColorOptionProvider";

export function ColorToggle() {
  const { option, setOption } = useColorOption();

  return (
    <div
      role="tablist"
      aria-label="Color option"
      className="inline-flex items-center rounded-full border border-hairline bg-muted/40 p-0.5 text-xs font-medium"
    >
      {(["vault", "marquee"] as const).map((id) => {
        const active = option === id;
        return (
          <button
            key={id}
            role="tab"
            aria-selected={active}
            onClick={() => setOption(id)}
            className={cn(
              "rounded-full px-3 py-1 transition-colors",
              active
                ? "bg-surface text-foreground shadow-card"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {COLOR_OPTIONS[id].name}
          </button>
        );
      })}
    </div>
  );
}
