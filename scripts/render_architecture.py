"""Render the current build's architecture diagram to a PNG.

Shows the layered architecture (UI → API → Orchestrator → Agents → Core/Data)
and how each agent integrates with the shared RunState, EventBus, LLM client,
tool layer, and per-run artifacts.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


W, H = 22, 14
fig, ax = plt.subplots(figsize=(W, H), dpi=160)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_axis_off()

# ----- colour palette -----
C_BG = "#fafafa"
C_UI = "#dbeafe"      # blue-100
C_UI_E = "#1d4ed8"
C_API = "#fef3c7"     # amber-100
C_API_E = "#b45309"
C_ORCH = "#e9d5ff"    # purple-200
C_ORCH_E = "#6b21a8"
C_AGENT_REAL = "#dcfce7"    # green-100
C_AGENT_REAL_E = "#15803d"
C_AGENT_STUB = "#fee2e2"    # red-100
C_AGENT_STUB_E = "#b91c1c"
C_CORE = "#f1f5f9"    # slate-100
C_CORE_E = "#334155"
C_LLM = "#ffe4e6"     # rose-100
C_LLM_E = "#9f1239"
C_DATA = "#cffafe"    # cyan-100
C_DATA_E = "#0e7490"
C_GATE = "#fde68a"
C_GATE_E = "#92400e"

ax.add_patch(Rectangle((0, 0), W, H, color=C_BG, zorder=-1))


def box(x, y, w, h, label, face, edge, *, fontsize=10, weight="bold", lh=1.2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=1.4, facecolor=face, edgecolor=edge,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize,
            color=edge, fontweight=weight, linespacing=lh)


def arrow(x1, y1, x2, y2, *, color="#475569", lw=1.4, style="->", text=None, dashed=False):
    ls = (0, (5, 4)) if dashed else "-"
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, linestyle=ls,
    )
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, text,
                ha="center", va="center", fontsize=8,
                color=color, backgroundcolor=C_BG)


# Title
ax.text(W / 2, H - 0.35,
        "Agentic Price & Promo Optimization — Architecture",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color="#0f172a")
ax.text(W / 2, H - 0.78,
        "14-agent DAG · FastAPI + SSE · Next.js · DuckDB+dbt+GE · Anthropic SDK (dry-run capable)",
        ha="center", va="center", fontsize=10, color="#475569")

# ----- 1. Web (top) -----
box(0.5, H - 2.4, 9, 1.2,
    "Next.js 14 (App Router, TS)\n"
    "• /runs · /runs/[id] · agent cards · ECharts inline visuals\n"
    "• Same-origin /api/* → rewrite proxy in next.config.mjs",
    C_UI, C_UI_E, fontsize=9, weight="normal")

box(10.0, H - 2.4, 4.0, 1.2,
    "CLI (Typer)\n"
    "uv run automl run | seed |\nbaseline-create",
    C_UI, C_UI_E, fontsize=9, weight="normal")

box(14.5, H - 2.4, 7.0, 1.2,
    "Browser ⇆ EventSource (SSE)\n"
    "agent_started · tool_called · agent_finished\n"
    "approval_required · approval_resolved",
    C_UI, C_UI_E, fontsize=9, weight="normal")

# ----- 2. FastAPI layer -----
box(0.5, H - 4.0, 21.0, 1.1,
    "FastAPI  (api/main.py)\n"
    "POST /runs   ·   GET /runs/{id}   ·   GET /runs/{id}/events (SSE)   ·"
    "   POST /runs/{id}/approve|reject   ·   POST /uploads   ·"
    "   GET /artifacts/{run_id}/{path}  (pathlib-safe)",
    C_API, C_API_E, fontsize=9.5, weight="normal")

# arrows UI → API
arrow(5.0, H - 2.4, 5.0, H - 2.9, color=C_UI_E)
arrow(12.0, H - 2.4, 12.0, H - 2.9, color=C_UI_E)
arrow(18.0, H - 2.4, 18.0, H - 2.9, color=C_UI_E, dashed=True)

# ----- 3. Orchestrator -----
box(0.5, H - 6.2, 12.0, 1.9,
    "Orchestrator   (core/orchestrator/)\n\n"
    "runner.execute(run, gates_enabled)  →  iterates AGENT_ORDER\n"
    "graph.py  ·  events.EventBus  ·  gates.GateRegistry  ·  state.RunState",
    C_ORCH, C_ORCH_E, fontsize=10, weight="normal")

box(13.0, H - 6.2, 4.5, 1.9,
    "RunState (Pydantic)\n"
    "• AgentResult per stage\n"
    "• ToolCall / ArtifactRef\n"
    "• tokens_in/out · cost_usd\n"
    "→ state.json + events.jsonl",
    C_ORCH, C_ORCH_E, fontsize=9, weight="normal")

box(18.0, H - 6.2, 3.5, 1.9,
    "Approval Gates\n"
    "default ON for:\n"
    "ppg_mapping · modeling\n"
    "· optimization\n"
    "(--no-gates skips)",
    C_GATE, C_GATE_E, fontsize=9, weight="normal")

arrow(11.0, H - 4.0, 11.0, H - 4.3, color=C_API_E)

# ----- 4. Agent DAG (the heart of the diagram) -----
ax.text(W / 2, H - 6.55, "Agent DAG  ·  AGENT_ORDER  (core/agents/, each subclasses Agent base)",
        ha="center", va="top", fontsize=11, fontweight="bold", color="#0f172a")

agents = [
    ("1. Ingestion", "real", "DuckDB + dbt build\n+ GE checkpoint"),
    ("2. PPG Mapping", "real", "cluster SKUs\n(gate)"),
    ("3. PPG Selection", "real", "coverage / sufficiency"),
    ("4. Feature Selection", "real", "candidate features"),
    ("5. EDA", "real", "distributions /\ntimeseries / lift"),
    ("6. Feature Eng.", "real", "logs · lags ·\nFourier · holidays"),
    ("7. Feature Refine", "real", "VIF · |corr|>0.95\n· MI · RFE"),
    ("8. Modeling", "stub", "log-log → semilog\n→ LightGBM → PyMC\n(gate)"),
    ("9. Results & Reason.", "stub", "MAPE/WAPE\n· sanity"),
    ("10. Decomposition", "stub", "base vs incremental\ndue-to"),
    ("11. Simulation", "stub", "price/promo\nscenario grids"),
    ("12. Optimization", "stub", "scipy → PuLP MILP\n(gate)"),
    ("13. Validation", "stub", "holdout · CV ·\nstability"),
    ("14. Insights & Report", "stub", "HTML + PDF"),
]

dag_top = H - 6.85
agent_w, agent_h = 1.45, 1.5
gap_x = 0.07
total = len(agents) * agent_w + (len(agents) - 1) * gap_x
x0 = (W - total) / 2

for i, (name, kind, sub) in enumerate(agents):
    x = x0 + i * (agent_w + gap_x)
    y = dag_top - agent_h
    if kind == "real":
        face, edge = C_AGENT_REAL, C_AGENT_REAL_E
        badge = "impl"
    else:
        face, edge = C_AGENT_STUB, C_AGENT_STUB_E
        badge = "stub"
    box(x, y, agent_w, agent_h, f"{name}\n\n{sub}", face, edge,
        fontsize=7.2, weight="normal", lh=1.15)
    ax.text(x + agent_w - 0.05, y + agent_h - 0.05, badge,
            ha="right", va="top", fontsize=6, color=edge,
            fontweight="bold")
    # gate marker on gated agents
    if "(gate)" in sub:
        ax.add_patch(FancyBboxPatch(
            (x + 0.05, y + 0.05), 0.35, 0.22,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=C_GATE, edgecolor=C_GATE_E, linewidth=1.0,
        ))
        ax.text(x + 0.225, y + 0.16, "G",
                ha="center", va="center", fontsize=7,
                color=C_GATE_E, fontweight="bold")

    # arrows between adjacent agents
    if i < len(agents) - 1:
        nx = x0 + (i + 1) * (agent_w + gap_x)
        arrow(x + agent_w, y + agent_h / 2, nx, y + agent_h / 2,
              color="#475569", lw=0.9)

# orchestrator → agents
arrow(6.5, H - 6.2, 6.5, dag_top - 0.02, color=C_ORCH_E, lw=1.4)

# legend strip under DAG
legend_y = dag_top - agent_h - 0.55
def legend_swatch(x, label, face, edge):
    ax.add_patch(FancyBboxPatch(
        (x, legend_y), 0.4, 0.28,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=face, edgecolor=edge, linewidth=1.0))
    ax.text(x + 0.5, legend_y + 0.14, label, ha="left", va="center",
            fontsize=8.5, color="#0f172a")

legend_swatch(2.5, "implemented (Phase 1+2)", C_AGENT_REAL, C_AGENT_REAL_E)
legend_swatch(7.0, "StubAgent (Phase 3+)", C_AGENT_STUB, C_AGENT_STUB_E)
legend_swatch(11.5, "G = approval gate", C_GATE, C_GATE_E)
legend_swatch(15.5, "dashed = event stream", "#ffffff", "#475569")

# ----- 5. Agent integration (zoom into one agent's wiring) -----
section_y = legend_y - 0.4
ax.text(W / 2, section_y, "How each agent integrates  (Agent base class)",
        ha="center", va="top", fontsize=11, fontweight="bold", color="#0f172a")

ag_y = section_y - 2.3
# inner agent box
box(0.5, ag_y, 5.0, 2.1,
    "Agent  (core/agents/base.py)\n\n"
    "async _execute(run, result)\n"
    "result.outputs / reasoning / confidence\n"
    "result.artifacts ← ArtifactRef[]\n"
    "await self.emit(run, 'tool_called', …)",
    C_AGENT_REAL, C_AGENT_REAL_E, fontsize=9, weight="normal", lh=1.25)

# LLM client
box(6.2, ag_y + 1.1, 4.6, 1.0,
    "AnthropicClient  (core/llm/client.py)\n"
    "• prompt caching · dry_run when no API key\n"
    "• routing.py → Opus 4.7 / Sonnet 4.6",
    C_LLM, C_LLM_E, fontsize=8.5, weight="normal")

# Tools / Core
box(6.2, ag_y, 4.6, 1.0,
    "Core / Tools\n"
    "core/data · ppg · features · eda · charts\n"
    "(pure-Python, Pydantic-validated)",
    C_CORE, C_CORE_E, fontsize=8.5, weight="normal")

# Data layer
box(11.5, ag_y, 5.0, 2.1,
    "Per-run data  (runs/<id>/)\n\n"
    "warehouse.duckdb (dbt main.panel)\n"
    "ingestion_report.json\n"
    "*.json  chart-ready artefacts\n"
    "<agent>_llm_trace.json\n"
    "state.json · events.jsonl",
    C_DATA, C_DATA_E, fontsize=9, weight="normal", lh=1.25)

# Quality contract
box(17.0, ag_y + 1.1, 4.5, 1.0,
    "dbt project  (dbt/automl_dbt/)\n"
    "sources + staging + panel mart\n"
    "schema + singular tests",
    C_DATA, C_DATA_E, fontsize=8.5, weight="normal")
box(17.0, ag_y, 4.5, 1.0,
    "Great Expectations 1.x\n"
    "volume · distribution · relationship\n"
    "drift (baseline JSON) · anomalies",
    C_DATA, C_DATA_E, fontsize=8.5, weight="normal")

# wiring arrows
arrow(5.5, ag_y + 1.55, 6.2, ag_y + 1.55, color=C_LLM_E, text="call_llm")
arrow(5.5, ag_y + 0.5, 6.2, ag_y + 0.5, color=C_CORE_E, text="tools")
arrow(10.8, ag_y + 1.55, 11.5, ag_y + 1.55, color=C_DATA_E, dashed=True)
arrow(10.8, ag_y + 0.5, 11.5, ag_y + 0.5, color=C_DATA_E, text="artefacts")
arrow(16.5, ag_y + 1.55, 17.0, ag_y + 1.55, color=C_DATA_E, dashed=True)
arrow(16.5, ag_y + 0.5, 17.0, ag_y + 0.5, color=C_DATA_E, dashed=True)

# emit → EventBus → SSE (loops up to UI)
arrow(3.0, ag_y + 2.1, 3.0, ag_y + 2.55, color="#475569", dashed=True,
      text="emit() → EventBus → SSE")

# RunState arrow (agents read/write RunState)
arrow(13.5, ag_y + 2.1, 13.5, ag_y + 2.55, color=C_ORCH_E, dashed=True,
      text="RunState save / load")

# footer / acceptance gates
ax.text(W / 2, 0.35,
        "Verification (per phase, progress.md):  "
        "P1 ≥95% PPG agreement   ·   P2 VIF<10 & |corr|<0.95   ·   "
        "P3 elasticity sign ≥7/8 PPGs   ·   P4 decomp reconciles   ·   "
        "P5 MILP respects ladder/margin/comp-gap   ·   P6 HTML+PDF + cost dashboard",
        ha="center", va="center", fontsize=8.5, color="#334155", style="italic")

from pathlib import Path

out = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight", facecolor=C_BG)
print(f"wrote {out}")
