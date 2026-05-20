"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { THEMES } from "@/lib/theme/tokens";
import { fmtUsdCompact } from "@/lib/format";
import type { Variant, WeeklyPoint } from "@/lib/types";

interface Props {
  variant: Variant;
  data: WeeklyPoint[];
}

export function RevenueChart({ variant, data }: Props) {
  const c = THEMES[variant].chartColors;
  return (
    <div className="h-[320px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 16, right: 12, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={`grad-base-${variant}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={c.secondary} stopOpacity={0.35} />
              <stop offset="95%" stopColor={c.secondary} stopOpacity={0} />
            </linearGradient>
            <linearGradient id={`grad-prop-${variant}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={c.primary} stopOpacity={0.55} />
              <stop offset="95%" stopColor={c.primary} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={c.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="week"
            stroke="currentColor"
            opacity={0.5}
            tickLine={false}
            axisLine={false}
            fontSize={12}
          />
          <YAxis
            stroke="currentColor"
            opacity={0.5}
            tickLine={false}
            axisLine={false}
            fontSize={12}
            tickFormatter={(v) => fmtUsdCompact(Number(v))}
            width={64}
          />
          <Tooltip
            cursor={{ stroke: c.primary, strokeOpacity: 0.4 }}
            contentStyle={{
              background: "hsl(var(--raised))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 8,
              fontSize: 12,
              color: "hsl(var(--foreground))",
            }}
            formatter={(v: number) => fmtUsdCompact(v)}
          />
          <Legend
            iconType="line"
            wrapperStyle={{ fontSize: 12, color: "hsl(var(--muted-foreground))" }}
          />
          <Area
            type="monotone"
            dataKey="baseline_revenue"
            name="Status quo"
            stroke={c.secondary}
            strokeWidth={2}
            fill={`url(#grad-base-${variant})`}
            isAnimationActive
          />
          <Area
            type="monotone"
            dataKey="proposed_revenue"
            name="Proposed"
            stroke={c.primary}
            strokeWidth={2.5}
            fill={`url(#grad-prop-${variant})`}
            isAnimationActive
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
