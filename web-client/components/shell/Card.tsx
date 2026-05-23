import * as React from "react";

import { cn } from "@/lib/cn";

export const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "rounded-md border border-hairline bg-surface",
        className,
      )}
      {...props}
    />
  ),
);
Card.displayName = "Card";

export const RaisedCard = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "rounded-md border border-hairline bg-raised shadow-card",
        className,
      )}
      {...props}
    />
  ),
);
RaisedCard.displayName = "RaisedCard";

export const AnchorCard = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "rounded-lg border border-hairline bg-surface shadow-raised",
        className,
      )}
      {...props}
    />
  ),
);
AnchorCard.displayName = "AnchorCard";
