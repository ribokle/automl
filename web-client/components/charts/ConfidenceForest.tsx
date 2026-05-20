"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { THEMES } from "@/lib/theme/tokens";
import type { PPGForestPoint, Variant } from "@/lib/types";

interface Props {
  variant: Variant;
  data: PPGForestPoint[];
}

export function ConfidenceForest({ variant, data }: Props) {
  const c = THEMES[variant].chartColors;
  const shaped = data.map((p) => ({
    ppg: p.ppg_id.replace("ppg_", ""),
    elasticity: p.elasticity,
    err: [Math.abs(p.elasticity - p.ci_low), Math.abs(p.ci_high - p.elasticity)] as [
      number,
      number,
    ],
  }));

  return (
    <div className="h-[320px] w-full">
      <ResponsiveContainer>
        <BarChart
          data={shaped}
          layout="vertical"
          margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
        >
          <CartesianGrid stroke={c.grid} strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            stroke="currentColor"
            opacity={0.5}
            tickLine={false}
            axisLine={false}
            fontSize={12}
          />
          <YAxis
            type="category"
            dataKey="ppg"
            stroke="currentColor"
            opacity={0.6}
            tickLine={false}
            axisLine={false}
            fontSize={12}
            width={40}
            tickFormatter={(v) => `PPG ${v}`}
          />
          <Tooltip
            cursor={{ fill: "hsl(var(--muted) / 0.3)" }}
            contentStyle={{
              background: "hsl(var(--raised))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 8,
              fontSize: 12,
              color: "hsl(var(--foreground))",
            }}
            formatter={(v: number) => v.toFixed(2)}
          />
          <ReferenceLine x={0} stroke="currentColor" strokeOpacity={0.4} />
          <ReferenceLine x={-1} stroke={c.primary} strokeDasharray="3 3" strokeOpacity={0.4} />
          <Bar dataKey="elasticity" radius={[0, 6, 6, 0]} barSize={14}>
            {shaped.map((d, i) => (
              <Cell key={i} fill={d.elasticity < -1 ? c.primary : c.secondary} />
            ))}
            <ErrorBar dataKey="err" width={6} strokeWidth={1.5} stroke="currentColor" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
