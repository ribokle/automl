"""Async DAG executor.

Iterates `AGENT_ORDER`, delegating each stage to an `Agent` subclass. When an
agent name is listed in `run.gates`, the runner pauses after that agent runs
and waits on a `GateRegistry` event before continuing. Phase 1 implements the
three data-preparation agents for real and leaves later stages as stubs.
"""
from __future__ import annotations

from core.agents.base import Agent, StubAgent
from core.llm.client import AnthropicClient, LLMProvider
from core.agents.advanced_eda import AdvancedEDAAgent
from core.agents.decomposition import DecompositionAgent
from core.agents.eda import EDAAgent
from core.agents.feature_engineering import FeatureEngineeringAgent
from core.agents.feature_refine import FeatureRefineAgent
from core.agents.feature_selection import FeatureSelectionAgent
from core.agents.ingestion import IngestionAgent
from core.agents.insights import InsightsAgent
from core.agents.modeling import ModelingAgent
from core.agents.optimization import OptimizationAgent
from core.agents.ppg_mapping import PPGMappingAgent
from core.agents.ppg_selection import PPGSelectionAgent
from core.agents.results_reasoning import ResultsReasoningAgent
from core.agents.simulation import SimulationAgent
from core.agents.validation import ValidationAgent
from core.orchestrator.events import bus
from core.orchestrator.gates import DEFAULT_GATES, gate_registry
from core.orchestrator.state import AGENT_ORDER, AgentStatus, RunState, RunStatus


REAL_AGENTS: dict[str, type[Agent]] = {
    "ingestion": IngestionAgent,
    "ppg_mapping": PPGMappingAgent,
    "ppg_selection": PPGSelectionAgent,
    "feature_selection": FeatureSelectionAgent,
    "eda": EDAAgent,
    "advanced_eda": AdvancedEDAAgent,
    "feature_engineering": FeatureEngineeringAgent,
    "feature_refine": FeatureRefineAgent,
    "modeling": ModelingAgent,
    "results_reasoning": ResultsReasoningAgent,
    "decomposition": DecompositionAgent,
    "simulation": SimulationAgent,
    "optimization": OptimizationAgent,
    "validation": ValidationAgent,
    "insights": InsightsAgent,
}


def _build_agent(name: str, *, agent_mode: bool = True) -> Agent:
    cls = REAL_AGENTS.get(name)
    llm = AnthropicClient() if agent_mode else AnthropicClient(provider=LLMProvider.DRY_RUN)
    return cls(llm=llm) if cls else StubAgent(name=name, llm=llm)


async def _wait_for_gate(run: RunState, agent_name: str) -> bool:
    """Pause until approve / reject; loop on rerun signals.

    A `/rerun` resolution merges new ``run.options`` overrides, re-runs the
    agent, then re-arms the gate so the user can review the new output. The
    loop exits once the user approves or rejects.
    """
    if not run.gates.get(agent_name):
        return True
    while True:
        state = gate_registry.get(run.id, agent_name)
        prior_status = run.agents[agent_name].status
        run.agents[agent_name].status = AgentStatus.awaiting_approval
        run.status = RunStatus.awaiting_approval
        run.save()
        await bus.publish(
            run.id,
            run.run_dir,
            {"type": "approval_required", "agent": agent_name},
        )
        await state.event.wait()

        if state.rerun_payload is not None:
            payload = state.rerun_payload
            run.options = {**run.options, agent_name: {**run.options.get(agent_name, {}), **payload}}
            run.save()
            await bus.publish(
                run.id,
                run.run_dir,
                {"type": "agent_rerunning", "agent": agent_name, "options": payload},
            )
            gate_registry.reset(run.id, agent_name)
            run.agents[agent_name].status = AgentStatus.pending
            run.agents[agent_name].error = None
            run.agents[agent_name].artifacts = []
            run.agents[agent_name].outputs = {}
            try:
                agent_mode = bool(run.options.get("agent_mode", True))
                await _build_agent(agent_name, agent_mode=agent_mode).run(run)
                run.save()
            except Exception as exc:  # noqa: BLE001
                run.agents[agent_name].status = AgentStatus.failed
                run.agents[agent_name].error = str(exc)
                run.status = RunStatus.failed
                run.save()
                await bus.publish(
                    run.id,
                    run.run_dir,
                    {"type": "agent_failed", "agent": agent_name, "error": str(exc)},
                )
                return False
            continue

        approved = bool(state.approved)
        # Approval may carry a top-level config payload (e.g. the grain
        # selector at ppg_mapping). Merge into run.options BEFORE
        # downstream agents start so feature_engineering / modelling
        # see the right grain.
        if approved and state.approve_payload:
            payload = state.approve_payload
            for key, value in payload.items():
                run.options[key] = value
            await bus.publish(
                run.id,
                run.run_dir,
                {
                    "type": "approval_payload",
                    "agent": agent_name,
                    "options": payload,
                },
            )
        run.agents[agent_name].status = prior_status if approved else AgentStatus.failed
        run.status = RunStatus.running if approved else RunStatus.failed
        run.save()
        await bus.publish(
            run.id,
            run.run_dir,
            {"type": "approval_resolved", "agent": agent_name, "approved": approved},
        )
        return approved


async def execute(
    run: RunState,
    gates_enabled: bool = True,
    agent_mode: bool = True,
    grain_gate_required: bool = False,
) -> RunState:
    run.status = RunStatus.running
    if gates_enabled:
        run.gates = dict(DEFAULT_GATES)
    else:
        run.gates = {}
    # The grain-selector flow always pauses at ppg_mapping regardless of
    # gates_enabled, so the UI can render the selector even on
    # otherwise-headless runs.
    if grain_gate_required:
        run.gates["ppg_mapping"] = True
        run.options = {**run.options, "grain_gate_required": True}
    run.options = {**run.options, "agent_mode": agent_mode}
    run.save()

    await bus.publish(run.id, run.run_dir, {"type": "run_started", "agents": AGENT_ORDER})

    for agent_name in AGENT_ORDER:
        agent = _build_agent(agent_name, agent_mode=agent_mode)
        try:
            await agent.run(run)
            run.save()
        except Exception as exc:  # noqa: BLE001
            run.agents[agent_name].status = AgentStatus.failed
            run.agents[agent_name].error = str(exc)
            run.status = RunStatus.failed
            run.save()
            await bus.publish(
                run.id,
                run.run_dir,
                {"type": "agent_failed", "agent": agent_name, "error": str(exc)},
            )
            break

        if run.agents[agent_name].status == AgentStatus.failed:
            run.status = RunStatus.failed
            run.save()
            break

        # Wait at gates configured for this run.
        if agent_name in run.gates:
            approved = await _wait_for_gate(run, agent_name)
            if not approved:
                run.status = RunStatus.failed
                run.agents[agent_name].error = "rejected at approval gate"
                run.save()
                await bus.publish(
                    run.id,
                    run.run_dir,
                    {"type": "run_finished", "status": "failed", "reason": "gate_rejected"},
                )
                gate_registry.drop(run.id)
                return run

    # Multi-grain comparison fan-out. Picks up any extra grains the
    # operator selected at the ppg_mapping gate and re-runs ONLY
    # feature_engineering + modeling for each so downstream
    # (decomposition / validation / optimisation) stays locked to the
    # primary grain. Per-grain artifacts get a double-underscore
    # suffix so they don't collide with the primary outputs.
    if run.status != RunStatus.failed:
        await _run_comparison_grains(run, agent_mode=agent_mode)

    if run.status != RunStatus.failed:
        run.status = RunStatus.completed
    run.save()
    await bus.publish(run.id, run.run_dir, {"type": "run_finished", "status": run.status.value})
    gate_registry.drop(run.id)
    return run


_COMPARISON_SNAPSHOT_FILES = (
    "modeling_results.json",
    "elasticity_per_ppg.json",
    "elasticity_per_ppg_pooled.json",
    "modeling_preflight.json",
)


async def _run_comparison_grains(run: RunState, *, agent_mode: bool) -> None:
    """For each grain in ``run.options["comparison_grains"]``, run a
    fresh feature_engineering + modeling pass and write per-grain
    artifacts (``modeling_results__<grain>.json``,
    ``elasticity_per_ppg__<grain>.json``, ...).

    Snapshots the primary modelling outputs before the loop and
    restores them at the end so the primary artifacts stay at their
    canonical filenames (downstream agents and the UI's default tab
    both consume those).

    Failures here are non-fatal: a comparison miss shouldn't tear down
    a healthy primary run. The event stream surfaces them so the UI
    can show which comparisons completed.
    """
    import shutil
    from pathlib import Path

    grains = run.options.get("comparison_grains") or []
    if not grains:
        return

    primary_grain = run.options.get("modelling_grain") or "ppg_week"
    comparisons = [g for g in grains if g != primary_grain]
    if not comparisons:
        return

    run_dir = Path(run.run_dir)
    # Snapshot the primary modelling artifacts BEFORE the loop so we
    # can restore them once comparisons finish overwriting the
    # canonical filenames. Also snapshot the FE / modelling AgentResult
    # objects (outputs / artifacts / reasoning) — without this the
    # last-comparison's status would shadow the primary's in
    # state.json.
    snapshots: dict[Path, Path] = {}
    for fname in _COMPARISON_SNAPSHOT_FILES:
        src = run_dir / fname
        if src.exists():
            backup = src.with_name(f"{src.stem}.__primary_backup__{src.suffix}")
            shutil.copy2(src, backup)
            snapshots[src] = backup
    primary_results = {
        name: run.agents[name].model_copy(deep=True)
        for name in ("feature_engineering", "modeling")
        if name in run.agents
    }

    await bus.publish(
        run.id,
        run.run_dir,
        {
            "type": "comparison_started",
            "grains": list(comparisons),
            "primary": primary_grain,
        },
    )

    for grain in comparisons:
        await bus.publish(
            run.id,
            run.run_dir,
            {"type": "comparison_progress", "grain": grain, "stage": "feature_engineering"},
        )
        # Swap the primary modelling_grain into options just for this
        # pass; the FE / modelling agents read it from there.
        run.options["modelling_grain"] = grain
        run.save()
        try:
            await _build_agent("feature_engineering", agent_mode=agent_mode).run(run)
            await bus.publish(
                run.id,
                run.run_dir,
                {"type": "comparison_progress", "grain": grain, "stage": "modeling"},
            )
            await _build_agent("modeling", agent_mode=agent_mode).run(run)
        except Exception as exc:  # noqa: BLE001
            await bus.publish(
                run.id,
                run.run_dir,
                {
                    "type": "comparison_finished",
                    "grain": grain,
                    "status": "failed",
                    "error": str(exc),
                },
            )
            continue

        # Capture the comparison outputs under grain-suffixed names so
        # the next iteration can overwrite the canonical filenames freely.
        for fname in _COMPARISON_SNAPSHOT_FILES:
            src = run_dir / fname
            if not src.exists():
                continue
            dst = run_dir / f"{src.stem}__{grain}{src.suffix}"
            src.rename(dst)
        await bus.publish(
            run.id,
            run.run_dir,
            {"type": "comparison_finished", "grain": grain, "status": "done"},
        )

    # Restore the primary outputs and AgentResults.
    for canonical, backup in snapshots.items():
        if backup.exists():
            shutil.move(str(backup), str(canonical))
    for name, primary in primary_results.items():
        run.agents[name] = primary
    run.options["modelling_grain"] = primary_grain
    run.save()
