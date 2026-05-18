"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  CORR_LABELS,
  CORR_MATRIX,
  COVERAGE,
  PPGS,
  PPG_PRICE_BOX,
  SKU_SCATTER,
  TREND,
  VIF,
} from "../_data";

const TICK_STYLE = { fill: "#94a3b8", fontSize: 10 };

export function TrendChart({ height = 220 }: { height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={TREND} margin={{ top: 10, right: 12, bottom: 0, left: -20 }}>
        <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
        <XAxis dataKey="week" tick={TICK_STYLE} />
        <YAxis yAxisId="L" tick={TICK_STYLE} />
        <YAxis yAxisId="R" orientation="right" tick={TICK_STYLE} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 11 }} />
        <Line yAxisId="L" dataKey="units" stroke="#34d399" strokeWidth={2} dot={false} />
        <Line yAxisId="R" dataKey="price" stroke="#fbbf24" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PPGScatter({ height = 260 }: { height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 10, right: 20, bottom: 8, left: -8 }}>
        <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
        <XAxis type="number" dataKey="x" name="price tier" domain={[-0.4, 2.4]} tick={TICK_STYLE} />
        <YAxis type="number" dataKey="y" name="log price" domain={[0.2, 1.9]} tick={TICK_STYLE} />
        <Tooltip
          cursor={{ stroke: "#475569" }}
          contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 11 }}
        />
        {PPGS.map((p) => (
          <Scatter
            key={p.id}
            name={p.id}
            data={SKU_SCATTER.filter((s) => s.ppg === p.id)}
            fill={p.colour}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function VIFBar({ height = 240 }: { height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={VIF} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 60 }}>
        <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tick={TICK_STYLE} domain={[0, 10]} />
        <YAxis type="category" dataKey="feature" tick={{ ...TICK_STYLE, fontSize: 9 }} width={120} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 11 }} />
        <Bar dataKey="vif" fill="#34d399" radius={[0, 2, 2, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function BoxPlot() {
  const w = 360;
  const h = 200;
  const padX = 36;
  const padY = 18;
  const usable = w - padX * 2;
  const slot = usable / PPG_PRICE_BOX.length;
  const all = PPG_PRICE_BOX.flatMap((b) => [b.min, b.max]);
  const yMin = Math.min(...all);
  const yMax = Math.max(...all);
  const yScale = (v: number) => padY + (h - padY * 2) * (1 - (v - yMin) / (yMax - yMin));
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto">
      <rect x={0} y={0} width={w} height={h} fill="transparent" />
      {[yMin, (yMin + yMax) / 2, yMax].map((tick, i) => (
        <g key={i}>
          <line x1={padX} x2={w - padX} y1={yScale(tick)} y2={yScale(tick)} stroke="#1e293b" />
          <text x={padX - 4} y={yScale(tick) + 3} textAnchor="end" fontSize="9" fill="#94a3b8">
            ${tick.toFixed(2)}
          </text>
        </g>
      ))}
      {PPG_PRICE_BOX.map((b, i) => {
        const cx = padX + slot * (i + 0.5);
        const boxW = Math.min(28, slot * 0.55);
        return (
          <g key={b.ppg}>
            <line x1={cx} x2={cx} y1={yScale(b.min)} y2={yScale(b.max)} stroke={b.colour} strokeWidth={1.2} />
            <rect
              x={cx - boxW / 2}
              y={yScale(b.q3)}
              width={boxW}
              height={yScale(b.q1) - yScale(b.q3)}
              fill={b.colour}
              fillOpacity={0.25}
              stroke={b.colour}
            />
            <line
              x1={cx - boxW / 2}
              x2={cx + boxW / 2}
              y1={yScale(b.median)}
              y2={yScale(b.median)}
              stroke={b.colour}
              strokeWidth={2}
            />
            <text x={cx} y={h - 4} textAnchor="middle" fontSize="9" fill="#94a3b8">
              {b.ppg}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function CorrHeatmap() {
  const n = CORR_LABELS.length;
  const cell = 22;
  const padL = 100;
  const padT = 88;
  const w = padL + cell * n + 8;
  const h = padT + cell * n + 8;
  const colorFor = (r: number) => {
    const t = Math.max(-1, Math.min(1, r));
    if (t >= 0) {
      const a = Math.round(t * 255);
      return `rgba(52,211,153,${(0.2 + Math.abs(t) * 0.8).toFixed(2)})`;
    } else {
      return `rgba(244,114,182,${(0.2 + Math.abs(t) * 0.8).toFixed(2)})`;
    }
  };
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto">
      {CORR_LABELS.map((l, i) => (
        <text
          key={"r" + i}
          x={padL - 6}
          y={padT + cell * (i + 0.65)}
          textAnchor="end"
          fontSize="9"
          fill="#94a3b8"
        >
          {l}
        </text>
      ))}
      {CORR_LABELS.map((l, j) => (
        <text
          key={"c" + j}
          x={padL + cell * (j + 0.5)}
          y={padT - 8}
          textAnchor="end"
          transform={`rotate(-50 ${padL + cell * (j + 0.5)} ${padT - 8})`}
          fontSize="9"
          fill="#94a3b8"
        >
          {l}
        </text>
      ))}
      {CORR_MATRIX.map((row, i) =>
        row.map((r, j) => (
          <g key={`${i}-${j}`}>
            <rect
              x={padL + cell * j}
              y={padT + cell * i}
              width={cell - 1}
              height={cell - 1}
              fill={colorFor(r)}
            />
            <text
              x={padL + cell * (j + 0.5)}
              y={padT + cell * (i + 0.65)}
              textAnchor="middle"
              fontSize="8"
              fill="#0f172a"
            >
              {Math.abs(r) > 0.3 ? r.toFixed(2) : ""}
            </text>
          </g>
        )),
      )}
    </svg>
  );
}

export function CoverageGrid() {
  const cell = 12;
  const padL = 80;
  const padT = 12;
  const rows = COVERAGE.length;
  const cols = COVERAGE[0].cells.length;
  const w = padL + cell * cols + 6;
  const h = padT + cell * rows + 16;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto">
      {COVERAGE.map((r, i) => (
        <text
          key={r.sku}
          x={padL - 4}
          y={padT + cell * (i + 0.75)}
          textAnchor="end"
          fontSize="8"
          fill="#64748b"
        >
          {r.sku}
        </text>
      ))}
      {COVERAGE.map((r, i) =>
        r.cells.map((c, j) => (
          <rect
            key={`${i}-${j}`}
            x={padL + cell * j}
            y={padT + cell * i}
            width={cell - 1}
            height={cell - 1}
            fill={c ? "#34d399" : "#f43f5e"}
            fillOpacity={c ? 0.6 : 0.65}
          />
        )),
      )}
    </svg>
  );
}
