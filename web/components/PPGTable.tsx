"use client";

import { useEffect, useMemo, useState } from "react";

import { approveAgent, getPPGMappingTable, getPPGSelection, rejectAgent } from "@/lib/api";
import type { PPGRow, PPGSelectionRow, RunEvent } from "@/lib/types";

interface Props {
  runId: string;
  events: RunEvent[];
}

type ApprovalState = "idle" | "awaiting" | "approved" | "rejected" | "submitting";

function confidenceColor(conf: number): string {
  if (conf >= 0.9) return "bg-emerald-900/40 text-emerald-200";
  if (conf >= 0.7) return "bg-amber-900/40 text-amber-200";
  return "bg-rose-900/40 text-rose-200";
}

export function PPGTable({ runId, events }: Props) {
  const [rows, setRows] = useState<PPGRow[] | null>(null);
  const [selection, setSelection] = useState<PPGSelectionRow[] | null>(null);
  const [approval, setApproval] = useState<ApprovalState>("idle");
  const [pendingAgent, setPendingAgent] = useState<string | null>(null);

  // Derive approval state from the event stream.
  useEffect(() => {
    for (const e of events) {
      if (e.type === "approval_required" && e.agent === "ppg_mapping") {
        setApproval((s) => (s === "approved" || s === "rejected" ? s : "awaiting"));
        setPendingAgent("ppg_mapping");
      }
      if (e.type === "approval_resolved" && e.agent === "ppg_mapping") {
        setApproval(e.approved ? "approved" : "rejected");
        setPendingAgent(null);
      }
    }
  }, [events]);

  // Re-fetch the mapping table whenever ppg_mapping finishes.
  const mappingFinishedTs = useMemo(() => {
    const last = [...events].reverse().find(
      (e) => e.type === "agent_finished" && e.agent === "ppg_mapping",
    );
    return last?.ts ?? null;
  }, [events]);

  const selectionFinishedTs = useMemo(() => {
    const last = [...events].reverse().find(
      (e) => e.type === "agent_finished" && e.agent === "ppg_selection",
    );
    return last?.ts ?? null;
  }, [events]);

  useEffect(() => {
    if (!mappingFinishedTs) return;
    let cancelled = false;
    getPPGMappingTable(runId)
      .then((r) => {
        if (!cancelled) setRows(r);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [runId, mappingFinishedTs]);

  useEffect(() => {
    if (!selectionFinishedTs) return;
    let cancelled = false;
    getPPGSelection(runId)
      .then((r) => {
        if (!cancelled) setSelection(r);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [runId, selectionFinishedTs]);

  async function handleApprove() {
    if (!pendingAgent) return;
    setApproval("submitting");
    try {
      await approveAgent(runId, pendingAgent);
      setApproval("approved");
    } catch {
      setApproval("awaiting");
    }
  }

  async function handleReject() {
    if (!pendingAgent) return;
    setApproval("submitting");
    try {
      await rejectAgent(runId, pendingAgent);
      setApproval("rejected");
    } catch {
      setApproval("awaiting");
    }
  }

  if (!rows) return null;

  const grouped = new Map<string, PPGRow[]>();
  for (const r of rows) {
    if (!grouped.has(r.ppg_id)) grouped.set(r.ppg_id, []);
    grouped.get(r.ppg_id)!.push(r);
  }
  const selectionById = new Map<string, PPGSelectionRow>(
    (selection ?? []).map((s) => [s.ppg_id, s]),
  );

  return (
    <section className="mt-6 rounded-lg border border-slate-800 bg-slate-950/40 p-4">
      <header className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Price-Pack Groups</h2>
          <p className="text-xs text-slate-400">
            {grouped.size} PPGs over {rows.length} SKUs - mean confidence{" "}
            {(rows.reduce((s, r) => s + r.confidence, 0) / rows.length).toFixed(2)}
          </p>
        </div>
        <ApprovalControls
          state={approval}
          onApprove={handleApprove}
          onReject={handleReject}
        />
      </header>
      <div className="space-y-3">
        {Array.from(grouped.entries()).map(([ppg_id, members]) => {
          const sel = selectionById.get(ppg_id);
          const meanConf = members.reduce((s, r) => s + r.confidence, 0) / members.length;
          const rationale = members[0]?.rationale ?? "";
          const flagged = members.some((m) => m.flagged);
          return (
            <details key={ppg_id} className="rounded border border-slate-800 bg-slate-900/50" open>
              <summary className="cursor-pointer list-none px-3 py-2 text-sm">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-slate-200">{ppg_id}</span>
                    <span className="text-slate-400">
                      {members[0].brand} / {members[0].category}
                    </span>
                    <span className="text-xs text-slate-500">({members.length} SKUs)</span>
                    {flagged && (
                      <span className="rounded bg-amber-900/40 px-1.5 py-0.5 text-xs text-amber-200">
                        flagged
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-xs">
                    <span className={`rounded px-1.5 py-0.5 ${confidenceColor(meanConf)}`}>
                      conf {meanConf.toFixed(2)}
                    </span>
                    {sel && (
                      <span
                        className={`rounded px-1.5 py-0.5 ${
                          sel.eligible
                            ? "bg-emerald-900/40 text-emerald-200"
                            : "bg-slate-800 text-slate-300"
                        }`}
                      >
                        score {sel.score.toFixed(2)}{sel.eligible ? " · eligible" : " · held"}
                      </span>
                    )}
                  </div>
                </div>
                {rationale && (
                  <p className="mt-1 text-xs text-slate-400 italic">{rationale}</p>
                )}
              </summary>
              <table className="w-full text-xs">
                <thead className="bg-slate-900 text-slate-400">
                  <tr>
                    <th className="px-3 py-1 text-left font-medium">SKU</th>
                    <th className="px-3 py-1 text-left font-medium">Pack</th>
                    <th className="px-3 py-1 text-right font-medium">Median price</th>
                    <th className="px-3 py-1 text-right font-medium">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {members.map((m) => (
                    <tr key={m.sku} className="border-t border-slate-800">
                      <td className="px-3 py-1 font-mono">{m.sku}</td>
                      <td className="px-3 py-1">{m.pack_size}</td>
                      <td className="px-3 py-1 text-right tabular-nums">
                        ${m.median_price.toFixed(2)}
                      </td>
                      <td className="px-3 py-1 text-right tabular-nums">
                        <span className={`rounded px-1 py-0.5 ${confidenceColor(m.confidence)}`}>
                          {m.confidence.toFixed(2)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          );
        })}
      </div>
    </section>
  );
}

function ApprovalControls({
  state,
  onApprove,
  onReject,
}: {
  state: ApprovalState;
  onApprove: () => void;
  onReject: () => void;
}) {
  if (state === "approved") {
    return <span className="rounded bg-emerald-900/40 px-2 py-1 text-xs text-emerald-200">approved</span>;
  }
  if (state === "rejected") {
    return <span className="rounded bg-rose-900/40 px-2 py-1 text-xs text-rose-200">rejected</span>;
  }
  if (state === "awaiting" || state === "submitting") {
    const disabled = state === "submitting";
    return (
      <div className="flex gap-2">
        <button
          onClick={onApprove}
          disabled={disabled}
          className="rounded bg-emerald-700 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-600 disabled:opacity-50"
        >
          Approve mapping
        </button>
        <button
          onClick={onReject}
          disabled={disabled}
          className="rounded bg-slate-700 px-3 py-1 text-xs text-slate-200 hover:bg-slate-600 disabled:opacity-50"
        >
          Reject
        </button>
      </div>
    );
  }
  return null;
}
