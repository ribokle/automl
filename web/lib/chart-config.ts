// Shared ECharts defaults so every chart wraps the same baseline. Components
// can still override `height` per-instance via props; this file is the source
// of truth for grid margins, axis colours, and font sizes.

export const CHART_DEFAULTS = {
  height: 280,
  grid: { left: 56, right: 24, top: 32, bottom: 40 },
  fontSize: 10,
  textColor: "#cbd5e1",
  axisLine: "#475569",
};
