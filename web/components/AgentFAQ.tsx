"use client";

import { useState } from "react";

import { STAGE_FAQS } from "@/lib/agent-faqs";
import type { AgentName } from "@/lib/types";

interface Props {
  agent: AgentName;
}

export function AgentFAQ({ agent }: Props) {
  const entry = STAGE_FAQS[agent];
  const [open, setOpen] = useState(false);
  if (!entry || (!entry.questions.length && !entry.cornerCases.length)) return null;

  return (
    <div className="mt-3 rounded border border-slate-800 bg-slate-950/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-[11px] uppercase tracking-wider text-slate-400 hover:text-slate-200"
      >
        <span>Common questions &amp; corner cases</span>
        <span className="text-slate-500">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="space-y-4 border-t border-slate-800 px-3 py-3">
          {entry.questions.length > 0 && (
            <section>
              <div className="mb-2 text-[10px] uppercase tracking-wider text-slate-500">
                Client questions
              </div>
              <ul className="space-y-2">
                {entry.questions.map((f, i) => (
                  <li key={i} className="text-[11px]">
                    <p className="font-medium text-slate-200">Q: {f.q}</p>
                    <p className="mt-0.5 text-slate-400">A: {f.a}</p>
                  </li>
                ))}
              </ul>
            </section>
          )}
          {entry.cornerCases.length > 0 && (
            <section>
              <div className="mb-2 text-[10px] uppercase tracking-wider text-slate-500">
                Corner cases
              </div>
              <ul className="space-y-2">
                {entry.cornerCases.map((c, i) => (
                  <li key={i} className="text-[11px]">
                    <p className="text-slate-200">
                      <span className="font-mono text-slate-400">{c.condition}</span>{" "}
                      <span className="text-slate-500">→</span>{" "}
                      <span className="text-slate-300">{c.behaviour}</span>
                    </p>
                    {c.userSees && (
                      <p className="mt-0.5 pl-2 text-slate-500">User sees: {c.userSees}</p>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )}
          {entry.engineeringNotes.length > 0 && (
            <section>
              <div className="mb-2 text-[10px] uppercase tracking-wider text-slate-500">
                Engineering notes
              </div>
              <ul className="space-y-1 font-mono text-[11px] text-slate-400">
                {entry.engineeringNotes.map((n, i) => (
                  <li key={i}>
                    <span className="text-slate-300">{n.topic}</span>{" "}
                    <span className="text-slate-500">— {n.ref}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}
          <p className="text-[10px] text-slate-500">
            Full reference:{" "}
            <a
              href={`/docs/stage-faqs.md#${entry.docsAnchor}`}
              target="_blank"
              rel="noreferrer"
              className="text-emerald-400 hover:underline"
            >
              docs/stage-faqs.md#{entry.docsAnchor}
            </a>
          </p>
        </div>
      )}
    </div>
  );
}
