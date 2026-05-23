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

import { COLOR_OPTIONS } from "@/lib/theme/tokens";
import { fmtUsd, fmtIntCompact } from "@/lib/format";
import { useColorOption } from "@/components/shell/ColorOptionProvider";
import type { ElasticityPoint } from "@/lib/types";

interface Props {
  data: ElasticityPoint[];
  currentPrice: number;
  proposedPrice: number;
  metric: "units" | "revenue";
  height?: number;
}

export function ElasticityChart({
  data,
  currentPrice,
  proposedPrice,
  metric,
  height = 280,
}: Props) {
  const { option } = useColorOption();
  const c = COLOR_OPTIONS[option].chartColors;

  const findClosest = (price: number) =>
    data.reduce((best, p) => (Math.abs(p.price - price) < Math.abs(best.price - price) ? p : best));
  const currentPt = findClosest(currentPrice);
  const proposedPt = findClosest(proposedPrice);
  const yKey = metric === "revenue" ? "revenue" : "units";

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 16, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="price"
            stroke="currentColor"
            opacity={0.45}
            tickLine={false}
            axisLine={false}
            fontSize={11}
            tickFormatter={(v) => fmtUsd(Number(v))}
          />
          <YAxis
            stroke="currentColor"
            opacity={0.45}
            tickLine={false}
            axisLine={false}
            fontSize={11}
            width={56}
            tickFormatter={(v) =>
              metric === "revenue" ? fmtUsd(Number(v)) : fmtIntCompact(Number(v))
            }
          />
          <Tooltip
            cursor={{ stroke: c.primary, strokeOpacity: 0.5 }}
            contentStyle={{
              background: "hsl(var(--raised))",
              border: "1px solid hsl(var(--hairline))",
              borderRadius: 8,
              fontSize: 12,
              color: "hsl(var(--foreground))",
              boxShadow: "var(--shadow-card)",
            }}
            formatter={(v: number) =>
              metric === "revenue" ? fmtUsd(v) : fmtIntCompact(v)
            }
            labelFormatter={(p: number) => `Price ${fmtUsd(p)}`}
          />
          <Line
            type="monotone"
            dataKey={yKey}
            stroke={c.primary}
            strokeWidth={2.5}
            dot={false}
          />
          <ReferenceLine
            x={currentPt.price}
            stroke="currentColor"
            strokeOpacity={0.35}
            strokeDasharray="3 3"
            label={{ value: "Current", position: "top", fontSize: 10, fill: "currentColor" }}
          />
          <ReferenceLine
            x={proposedPt.price}
            stroke={c.primary}
            strokeDasharray="3 3"
            label={{ value: "You", position: "top", fontSize: 10, fill: c.primary }}
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
