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

import { COLOR_OPTIONS } from "@/lib/theme/tokens";
import { useColorOption } from "@/components/shell/ColorOptionProvider";
import type { PPGForestPoint } from "@/lib/types";

interface Props {
  data: PPGForestPoint[];
}

export function ConfidenceForest({ data }: Props) {
  const { option } = useColorOption();
  const c = COLOR_OPTIONS[option].chartColors;
  const shaped = data.map((p) => ({
    ppg: p.ppg_id.replace("ppg_", ""),
    elasticity: p.elasticity,
    err: [Math.abs(p.elasticity - p.ci_low), Math.abs(p.ci_high - p.elasticity)] as [
      number,
      number,
    ],
  }));

  return (
    <div className="h-[260px] w-full">
      <ResponsiveContainer>
        <BarChart
          data={shaped}
          layout="vertical"
          margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
        >
          <CartesianGrid stroke={c.grid} horizontal={false} />
          <XAxis
            type="number"
            stroke="currentColor"
            opacity={0.45}
            tickLine={false}
            axisLine={false}
            fontSize={11}
          />
          <YAxis
            type="category"
            dataKey="ppg"
            stroke="currentColor"
            opacity={0.55}
            tickLine={false}
            axisLine={false}
            fontSize={11}
            width={40}
            tickFormatter={(v) => `PPG ${v}`}
          />
          <Tooltip
            cursor={{ fill: "hsl(var(--muted) / 0.3)" }}
            contentStyle={{
              background: "hsl(var(--raised))",
              border: "1px solid hsl(var(--hairline))",
              borderRadius: 8,
              fontSize: 12,
              color: "hsl(var(--foreground))",
              boxShadow: "var(--shadow-card)",
            }}
            formatter={(v: number) => v.toFixed(2)}
          />
          <ReferenceLine x={0} stroke="currentColor" strokeOpacity={0.35} />
          <ReferenceLine x={-1} stroke={c.primary} strokeDasharray="3 3" strokeOpacity={0.45} />
          <Bar dataKey="elasticity" radius={[0, 4, 4, 0]} barSize={12}>
            {shaped.map((d, i) => (
              <Cell key={i} fill={d.elasticity < -1 ? c.primary : c.baseline} />
            ))}
            <ErrorBar dataKey="err" width={6} strokeWidth={1.2} stroke="currentColor" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
