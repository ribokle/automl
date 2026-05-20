"use client";

import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { THEMES } from "@/lib/theme/tokens";
import { fmtUsd, fmtIntCompact } from "@/lib/format";
import type { ElasticityPoint, Variant } from "@/lib/types";

interface Props {
  variant: Variant;
  data: ElasticityPoint[];
  currentPrice: number;
  proposedPrice: number;
  metric: "units" | "revenue";
}

export function ElasticityChart({ variant, data, currentPrice, proposedPrice, metric }: Props) {
  const c = THEMES[variant].chartColors;
  const findClosest = (price: number) =>
    data.reduce((best, p) => (Math.abs(p.price - price) < Math.abs(best.price - price) ? p : best));
  const currentPt = findClosest(currentPrice);
  const proposedPt = findClosest(proposedPrice);
  const yKey = metric === "revenue" ? "revenue" : "units";

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 16, right: 16, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={`elas-${variant}`} x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor={c.negative} stopOpacity={0.7} />
              <stop offset="100%" stopColor={c.primary} stopOpacity={0.9} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={c.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="price"
            stroke="currentColor"
            opacity={0.5}
            tickLine={false}
            axisLine={false}
            fontSize={12}
            tickFormatter={(v) => fmtUsd(Number(v))}
          />
          <YAxis
            stroke="currentColor"
            opacity={0.5}
            tickLine={false}
            axisLine={false}
            fontSize={12}
            width={64}
            tickFormatter={(v) =>
              metric === "revenue" ? fmtUsd(Number(v)) : fmtIntCompact(Number(v))
            }
          />
          <Tooltip
            cursor={{ stroke: c.primary, strokeOpacity: 0.5 }}
            contentStyle={{
              background: "hsl(var(--raised))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 8,
              fontSize: 12,
              color: "hsl(var(--foreground))",
            }}
            formatter={(v: number) =>
              metric === "revenue" ? fmtUsd(v) : fmtIntCompact(v)
            }
            labelFormatter={(p: number) => `Price ${fmtUsd(p)}`}
          />
          <Line
            type="monotone"
            dataKey={yKey}
            stroke={`url(#elas-${variant})`}
            strokeWidth={3}
            dot={false}
          />
          <ReferenceLine
            x={currentPt.price}
            stroke="currentColor"
            strokeOpacity={0.3}
            strokeDasharray="4 4"
            label={{ value: "Current", position: "top", fontSize: 10, fill: "currentColor" }}
          />
          <ReferenceLine
            x={proposedPt.price}
            stroke={c.primary}
            strokeDasharray="4 4"
            label={{ value: "Proposed", position: "top", fontSize: 10, fill: c.primary }}
          />
          <ReferenceDot
            x={proposedPt.price}
            y={proposedPt[yKey]}
            r={5}
            fill={c.primary}
            stroke="hsl(var(--surface))"
            strokeWidth={2}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
