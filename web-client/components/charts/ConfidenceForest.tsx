"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Label,
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

// Bijmolt, van Heerde & Pieters (2005) meta-analysis grand mean own-price
// elasticity across 1,851 studies. The reference line lets viewers eyeball
// "are we close to where the literature says we should be?" at a glance.
const BIJMOLT_GRAND_MEAN = -2.62;

type Shaped = {
  ppg: string;
  elasticity: number;
  err: [number, number];
  benchmark_status: NonNullable<PPGForestPoint["benchmark_status"]>;
  benchmark_low: number | null;
  benchmark_high: number | null;
  benchmark_mean: number | null;
};

export function ConfidenceForest({ data }: Props) {
  const { option } = useColorOption();
  const c = COLOR_OPTIONS[option].chartColors;
  const shaped: Shaped[] = data.map((p) => ({
    ppg: p.ppg_id.replace("ppg_", ""),
    elasticity: p.elasticity,
    err: [Math.abs(p.elasticity - p.ci_low), Math.abs(p.ci_high - p.elasticity)],
    benchmark_status: p.benchmark_status ?? "no_benchmark",
    benchmark_low: typeof p.benchmark_low === "number" ? p.benchmark_low : null,
    benchmark_high: typeof p.benchmark_high === "number" ? p.benchmark_high : null,
    benchmark_mean: typeof p.benchmark_mean === "number" ? p.benchmark_mean : null,
  }));

  return (
    <div className="h-[280px] w-full">
      <ResponsiveContainer>
        <BarChart
          data={shaped}
          layout="vertical"
          margin={{ top: 8, right: 24, left: 8, bottom: 24 }}
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
            formatter={(value: number, _name: string, item) => {
              const row = (item?.payload ?? {}) as Shaped;
              const band =
                row.benchmark_low != null && row.benchmark_high != null
                  ? `${row.benchmark_low.toFixed(2)} … ${row.benchmark_high.toFixed(2)}`
                  : "no benchmark";
              return [
                `${value.toFixed(2)} (band ${band})`,
                "Elasticity",
              ];
            }}
          />
          <ReferenceLine x={0} stroke="currentColor" strokeOpacity={0.35} />
          <ReferenceLine x={-1} stroke={c.primary} strokeDasharray="3 3" strokeOpacity={0.45} />
          <ReferenceLine
            x={BIJMOLT_GRAND_MEAN}
            stroke="currentColor"
            strokeDasharray="4 4"
            strokeOpacity={0.4}
          >
            <Label
              value="Bijmolt 2005 mean (-2.62)"
              position="insideBottomLeft"
              fill="currentColor"
              fontSize={10}
              opacity={0.6}
            />
          </ReferenceLine>
          <Bar dataKey="elasticity" radius={[0, 4, 4, 0]} barSize={12}>
            {shaped.map((d, i) => {
              const outOfBand =
                d.benchmark_status === "out_band_low" || d.benchmark_status === "out_band_high";
              return (
                <Cell
                  key={i}
                  fill={d.elasticity < -1 ? c.primary : c.baseline}
                  stroke={outOfBand ? "hsl(var(--warning))" : undefined}
                  strokeWidth={outOfBand ? 1.5 : 0}
                />
              );
            })}
            <ErrorBar dataKey="err" width={6} strokeWidth={1.2} stroke="currentColor" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
