import { cn } from "@/lib/cn";

interface Props {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  display?: boolean;
  className?: string;
  children?: React.ReactNode;
}

export function PageHeader({ eyebrow, title, subtitle, display, className, children }: Props) {
  return (
    <header
      className={cn(
        "flex flex-col gap-3 pb-6",
        display ? "pt-12" : "pt-8",
        className,
      )}
    >
      {eyebrow ? (
        <div className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
          {eyebrow}
        </div>
      ) : null}
      <h1
        className={cn(
          "font-display font-semibold tracking-tight text-balance",
          display ? "text-4xl md:text-5xl lg:text-6xl" : "text-2xl md:text-3xl",
        )}
      >
        {title}
      </h1>
      {subtitle ? (
        <p className={cn("max-w-3xl text-pretty text-muted-foreground", display ? "text-lg" : "text-base")}>
          {subtitle}
        </p>
      ) : null}
      {children}
    </header>
  );
}
