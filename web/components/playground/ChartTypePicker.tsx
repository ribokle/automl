"use client";

import { CHART_TYPE_LIST, type ChartTypeId, type ChartTypeMeta } from "./chart_types";

export function ChartTypePicker({
  value,
  onChange,
}: {
  value: ChartTypeId;
  onChange: (id: ChartTypeId) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {CHART_TYPE_LIST.map((t) => (
        <ChartTypeButton key={t.id} meta={t} active={t.id === value} onSelect={() => onChange(t.id)} />
      ))}
    </div>
  );
}

function ChartTypeButton({
  meta,
  active,
  onSelect,
}: {
  meta: ChartTypeMeta;
  active: boolean;
  onSelect: () => void;
}) {
  const tip = `${meta.blurb}\n\nDimensions: ${meta.dims.help}\nMeasures: ${meta.measures.help}`;
  return (
    <button
      type="button"
      onClick={onSelect}
      title={tip}
      className={`group relative rounded border px-3 py-2 text-left transition ${
        active
          ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-100"
          : "border-slate-700 bg-slate-900/60 text-slate-300 hover:border-slate-500 hover:bg-slate-800"
      }`}
    >
      <div className="text-[12px] font-semibold">{meta.label}</div>
      <div className="text-[10px] text-slate-400">{meta.blurb}</div>
      <div className="pointer-events-none absolute left-0 top-full z-30 mt-1 hidden w-64 rounded border border-slate-700 bg-slate-900 p-2 text-[10px] text-slate-300 shadow-lg group-hover:block group-focus:block">
        <div className="mb-1 font-semibold text-slate-100">{meta.label}</div>
        <div className="mb-1">
          <span className="text-slate-500">Dims:</span> {meta.dims.help}
        </div>
        <div>
          <span className="text-slate-500">Measures:</span> {meta.measures.help}
        </div>
      </div>
    </button>
  );
}
