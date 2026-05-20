"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { COLOR_OPTIONS } from "@/lib/theme/tokens";
import { fmtUsdCompact } from "@/lib/format";
import { useColorOption } from "@/components/shell/ColorOptionProvider";
import type { WeeklyPoint } from "@/lib/types";

interface Props {
  data: WeeklyPoint[];
}

export function RevenueChart({ data }: Props) {
  const { option } = useColorOption();
  const c = COLOR_OPTIONS[option].chartColors;

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 12, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={`grad-baseline-${option}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={c.secondary} stopOpacity={0.25} />
              <stop offset="100%" stopColor={c.secondary} stopOpacity={0} />
            </linearGradient>
            <linearGradient id={`grad-proposed-${option}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={c.primary} stopOpacity={0.45} />
              <stop offset="100%" stopColor={c.primary} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={c.grid} strokeDasharray="0" vertical={false} />
          <XAxis
            dataKey="week"
            stroke="currentColor"
            opacity={0.45}
            tickLine={false}
            axisLine={false}
            fontSize={11}
            interval="preserveStartEnd"
          />
          <YAxis
            stroke="currentColor"
            opacity={0.45}
            tickLine={false}
            axisLine={false}
            fontSize={11}
            tickFormatter={(v) => fmtUsdCompact(Number(v))}
            width={56}
          />
          <Tooltip
            cursor={{ stroke: c.primary, strokeOpacity: 0.4, strokeWidth: 1 }}
            contentStyle={{
              background: "hsl(var(--raised))",
              border: "1px solid hsl(var(--hairline))",
              borderRadius: 8,
              fontSize: 12,
              color: "hsl(var(--foreground))",
              boxShadow: "var(--shadow-card)",
            }}
            formatter={(v: number, name: string) => [fmtUsdCompact(v), name === "baseline_revenue" ? "Status quo" : "Proposed"]}
          />
          <Area
            type="monotone"
            dataKey="baseline_revenue"
            stroke={c.secondary}
            strokeWidth={1.5}
            strokeDasharray="3 3"
            fill={`url(#grad-baseline-${option})`}
            isAnimationActive
          />
          <Area
            type="monotone"
            dataKey="proposed_revenue"
            stroke={c.primary}
            strokeWidth={2}
            fill={`url(#grad-proposed-${option})`}
            isAnimationActive
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
