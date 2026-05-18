"use client";

import { useState } from "react";

export interface PPGTab {
  key: string;
  label: string;
  description: string;
  content: React.ReactNode;
}

export function PPGTabs({ tabs }: { tabs: PPGTab[] }) {
  const [active, setActive] = useState(tabs[0]?.key ?? "");
  const current = tabs.find((t) => t.key === active) ?? tabs[0];
  return (
    <div>
      <div className="mb-2 flex gap-1 border-b border-slate-800">
        {tabs.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setActive(t.key)}
            className={`-mb-px border-b-2 px-3 py-1.5 text-[11px] font-medium transition ${
              t.key === active
                ? "border-emerald-400 text-emerald-300"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      {current?.description && (
        <p className="mb-2 text-[10px] text-slate-500">{current.description}</p>
      )}
      <div>{current?.content}</div>
    </div>
  );
}
