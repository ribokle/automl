"""Async DAG executor.

Iterates `AGENT_ORDER`, delegating each stage to an `Agent` subclass. When an
agent name is listed in `run.gates`, the runner pauses after that agent runs
and waits on a `GateRegistry` event before continuing. Phase 1 implements the
three data-preparation agents for real and leaves later stages as stubs.
"""
from __future__ import annotations

from pathlib import Path

from core.agents.advanced_eda import AdvancedEDAAgent
from core.agents.base import Agent, StubAgent
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
from core.llm.client import AnthropicClient, LLMProvider
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
        try:
            await _run_comparison_grains(run, agent_mode=agent_mode)
        except Exception as exc:  # noqa: BLE001
            # The comparison fan-out is an enhancement on top of a
            # successful primary run; never let it tear down the run.
            await bus.publish(
                run.id,
                run.run_dir,
                {"type": "comparison_finished", "status": "failed", "error": str(exc)},
            )

    if run.status != RunStatus.failed:
        run.status = RunStatus.completed
    run.save()
    await bus.publish(run.id, run.run_dir, {"type": "run_finished", "status": run.status.value})
    gate_registry.drop(run.id)
    return run


# Tail of AGENT_ORDER that re-runs per comparison grain. Order matches
# AGENT_ORDER so each agent reads the previous one's canonical output.
# feature_engineering / feature_refine / results_reasoning are
# always-on prerequisites; the operator-facing "depth" picker (the
# UI's `comparison_agents` payload) decides where to stop.
_COMPARISON_DOWNSTREAM_AGENTS: tuple[str, ...] = (
    "feature_engineering",
    "feature_refine",
    "modeling",
    "results_reasoning",
    "decomposition",
    "simulation",
    "optimization",
    "validation",
    "insights",
)

# Operator-facing stages in the UI's depth picker. Picking a stage
# implicitly includes every stage above it (data dependency).
_COMPARISON_DEPTH_STAGES: tuple[str, ...] = (
    "modeling",
    "decomposition",
    "simulation",
    "optimization",
    "validation",
    "insights",
)


def _agents_for_depth(comparison_agents: list[str] | None) -> tuple[str, ...]:
    """Resolve the operator's depth picker into a concrete agent list.

    ``None`` means "not specified" → default to the full tail.
    An empty list means "explicitly nothing" → empty fan-out.
    Otherwise walk ``_COMPARISON_DOWNSTREAM_AGENTS`` and keep every
    agent up to and including the deepest stage the operator picked.
    feature_engineering / feature_refine / results_reasoning are
    always included as prerequisites when their dependents are.
    """
    if comparison_agents is None:
        return _COMPARISON_DOWNSTREAM_AGENTS
    if not comparison_agents:
        return ()
    requested = set(comparison_agents)
    # Find the deepest selected stage in canonical order.
    last_idx = -1
    for i, name in enumerate(_COMPARISON_DOWNSTREAM_AGENTS):
        if name in requested:
            last_idx = i
    if last_idx < 0:
        return ()
    return _COMPARISON_DOWNSTREAM_AGENTS[: last_idx + 1]


def _snapshot_mtimes(run_dir: "Path") -> dict[str, float]:
    """Capture ``{name: mtime}`` for every comparison artifact in ``run_dir``.

    Used as a baseline so the comparison loop can detect which files
    each grain's pass produced or overwrote, and rename only those.
    Restricted to JSON artifacts via :func:`_is_comparison_artifact` so
    run bookkeeping (``state.json``), the event log, and the DuckDB
    warehouse are never snapshotted, backed up, or renamed. The
    warehouse in particular is held open by the run's connection and is
    locked against copy on Windows.
    """
    return {
        p.name: p.stat().st_mtime
        for p in run_dir.iterdir()
        if p.is_file() and _is_comparison_artifact(p.name)
    }


# Run bookkeeping that lives in run_dir but is NOT a per-agent artifact.
# These must never be snapshotted / backed up / renamed by the fan-out.
_NON_ARTIFACT_FILES = frozenset({"state.json"})


def _is_comparison_artifact(name: str) -> bool:
    """True for per-agent JSON artifacts the fan-out may snapshot/rename.

    Excludes run state, the event log (``events.jsonl``), the DuckDB
    warehouse (``warehouse.duckdb`` and its WAL — locked on Windows),
    and any binary report. Downstream cards read JSON, so restricting to
    ``.json`` (minus ``state.json``) covers every per-grain panel while
    sidestepping the warehouse lock.
    """
    if name in _NON_ARTIFACT_FILES:
        return False
    return name.endswith(".json")


async def _run_comparison_grains(run: RunState, *, agent_mode: bool) -> None:
    """Re-run the downstream tail per comparison grain.

    For each grain in ``run.options["comparison_grains"]``, runs the
    agents picked by ``run.options["comparison_agents"]`` (default: the
    full ``_COMPARISON_DOWNSTREAM_AGENTS`` tail). Each pass writes to
    canonical artifact filenames so agents inside the loop read the
    previous stage's output naturally; at the end of each grain's pass
    we mtime-detect which files were touched and rename them to
    ``<stem>__<grain><suffix>``.

    Primary canonical outputs are backed up before the loop and
    restored after, so the primary-grain artifacts stay at their
    canonical names for the default UI tab and any post-hoc CLI tooling.

    Failures here are non-fatal: a comparison failure (per grain or per
    stage) emits a structured event and continues to the next grain so
    a single broken pass doesn't tear down a healthy primary run.
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

    requested_agents = run.options.get("comparison_agents")
    fanout_agents = _agents_for_depth(requested_agents)
    if not fanout_agents:
        return

    run_dir = Path(run.run_dir)

    # Baseline: capture which files exist + their mtimes BEFORE the loop.
    # Anything written during the loop will either be new (not in baseline)
    # or mtime-changed; that's the renamed-per-grain set. Anything not
    # touched stays where it is.
    baseline: dict[str, float] = _snapshot_mtimes(run_dir)

    # Back up canonical artifacts (basenames without "__") to siblings
    # so we can restore them after the comparison loop overwrites them.
    primary_backups: dict[Path, Path] = {}
    for name in baseline:
        if "__" in Path(name).stem:
            continue
        src = run_dir / name
        backup = src.with_name(f"{src.stem}.__primary_backup__{src.suffix}")
        try:
            shutil.copy2(src, backup)
        except OSError:
            # A locked or vanished file can't be backed up; skip it
            # rather than tearing down the whole comparison pass.
            continue
        primary_backups[src] = backup

    # Snapshot AgentResult state for every agent we're about to re-run
    # so the comparison's status doesn't shadow the primary in state.json.
    primary_results = {
        name: run.agents[name].model_copy(deep=True)
        for name in fanout_agents
        if name in run.agents
    }

    # Surface cost expectations once if we're not in dry-run.
    try:
        provider = run.options.get("llm_provider") or "dry_run"
    except Exception:  # noqa: BLE001
        provider = "dry_run"
    if provider != "dry_run":
        await bus.publish(
            run.id,
            run.run_dir,
            {
                "type": "comparison_cost_warning",
                "grains": list(comparisons),
                "agents": list(fanout_agents),
                "provider": provider,
            },
        )

    await bus.publish(
        run.id,
        run.run_dir,
        {
            "type": "comparison_started",
            "grains": list(comparisons),
            "primary": primary_grain,
            "agents": list(fanout_agents),
        },
    )

    total_agents = len(fanout_agents)
    try:
        for grain in comparisons:
            # Refresh the per-grain baseline so end-of-grain rename only
            # picks up files this grain wrote (not files written by the
            # previous comparison grain that already got renamed).
            per_grain_baseline = _snapshot_mtimes(run_dir)
            run.options["modelling_grain"] = grain
            run.save()
            failed_at: str | None = None
            for idx, agent_name in enumerate(fanout_agents):
                await bus.publish(
                    run.id,
                    run.run_dir,
                    {
                        "type": "comparison_progress",
                        "grain": grain,
                        "stage": agent_name,
                        "agent_index": idx,
                        "total_agents": total_agents,
                    },
                )
                try:
                    await _build_agent(agent_name, agent_mode=agent_mode).run(run)
                except Exception as exc:  # noqa: BLE001
                    failed_at = agent_name
                    await bus.publish(
                        run.id,
                        run.run_dir,
                        {
                            "type": "comparison_finished",
                            "grain": grain,
                            "status": "failed",
                            "failed_at": agent_name,
                            "error": str(exc),
                        },
                    )
                    break

            # Rename touched artifacts (new or mtime-changed) to the
            # per-grain suffix. Only JSON artifacts are eligible (see
            # _is_comparison_artifact) so state.json / events.jsonl /
            # warehouse.duckdb are never renamed. Skip basenames that
            # already contain "__" (other grains' artifacts, backup
            # markers) so repeated grains don't double-suffix.
            for fpath in list(run_dir.iterdir()):
                if not fpath.is_file():
                    continue
                if not _is_comparison_artifact(fpath.name):
                    continue
                if "__" in fpath.stem:
                    continue
                prior_mtime = per_grain_baseline.get(fpath.name)
                if prior_mtime is not None and fpath.stat().st_mtime <= prior_mtime:
                    continue
                dst = fpath.with_name(f"{fpath.stem}__{grain}{fpath.suffix}")
                try:
                    fpath.rename(dst)
                except OSError:
                    # Best-effort: if the target name is already taken
                    # (shouldn't happen in normal flow), leave the file in
                    # place rather than corrupting state.
                    pass

            if failed_at is None:
                await bus.publish(
                    run.id,
                    run.run_dir,
                    {"type": "comparison_finished", "grain": grain, "status": "done"},
                )
    finally:
        # Always restore the primary canonical artifacts, AgentResults,
        # and grain selection — even if the loop raised — so the primary
        # run's status and outputs are never left shadowed by a
        # comparison pass.
        for canonical, backup in primary_backups.items():
            if backup.exists():
                shutil.move(str(backup), str(canonical))
        for name, primary in primary_results.items():
            run.agents[name] = primary
        run.options["modelling_grain"] = primary_grain
        run.save()
