import { cn } from "@/lib/cn";

interface Props {
  label: string;
  value: string;
  delta?: string;
  detail?: string;
  positive?: boolean;
  size?: "lg" | "xl";
  className?: string;
}

export function AnchorNumber({
  label,
  value,
  delta,
  detail,
  positive = true,
  size = "lg",
  className,
}: Props) {
  const valueClass =
    size === "xl"
      ? "display text-[64px] sm:text-[80px] leading-[0.95] font-semibold tracking-tight tabular"
      : "display text-[56px] leading-[0.95] font-semibold tracking-tight tabular";

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      <div className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
        {label}
      </div>
      <div className="flex flex-wrap items-baseline gap-4">
        <span className={valueClass}>{value}</span>
        {delta ? (
          <span
            className={cn(
              "inline-flex items-center rounded-full border px-2.5 py-0.5 text-sm font-medium tabular",
              positive
                ? "border-positive/30 bg-positive/10 text-positive"
                : "border-negative/30 bg-negative/10 text-negative",
            )}
          >
            {delta}
          </span>
        ) : null}
      </div>
      {detail ? (
        <p className="max-w-2xl text-pretty text-sm text-muted-foreground">{detail}</p>
      ) : null}
    </div>
  );
}
