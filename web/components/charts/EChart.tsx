"use client";

import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import {
  BarChart,
  BoxplotChart,
  CustomChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
} from "echarts/charts";
import {
  DatasetComponent,
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsCoreOption, ECharts } from "echarts/core";

echarts.use([
  BarChart,
  BoxplotChart,
  CustomChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
  DatasetComponent,
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

export interface EChartProps {
  option: EChartsCoreOption;
  height?: number | string;
  className?: string;
  "data-chart"?: string;
}

export function EChart({ option, height = 240, className, ...rest }: EChartProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const inst = useRef<ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    inst.current = echarts.init(ref.current, undefined, { renderer: "canvas" });
    const handle = () => inst.current?.resize();
    window.addEventListener("resize", handle);
    return () => {
      window.removeEventListener("resize", handle);
      inst.current?.dispose();
      inst.current = null;
    };
  }, []);

  useEffect(() => {
    inst.current?.setOption(option, true);
  }, [option]);

  return (
    <div
      ref={ref}
      className={className}
      style={{ width: "100%", height }}
      {...rest}
    />
  );
}
