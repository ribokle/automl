"""Async DAG executor.

Iterates `AGENT_ORDER`, delegating each stage to an `Agent` subclass. When an
agent name is listed in `run.gates`, the runner pauses after that agent runs
and waits on a `GateRegistry` event before continuing. Phase 1 implements the
three data-preparation agents for real and leaves later stages as stubs.
"""
from __future__ import annotations

from core.agents.base import Agent, StubAgent
from core.agents.decomposition import DecompositionAgent
from core.agents.eda import EDAAgent
from core.agents.feature_engineering import FeatureEngineeringAgent
from core.agents.feature_refine import FeatureRefineAgent
from core.agents.feature_selection import FeatureSelectionAgent
from core.agents.ingestion import IngestionAgent
from core.agents.modeling import ModelingAgent
from core.agents.optimization import OptimizationAgent
from core.agents.ppg_mapping import PPGMappingAgent
from core.agents.ppg_selection import PPGSelectionAgent
from core.agents.results_reasoning import ResultsReasoningAgent
from core.agents.simulation import SimulationAgent
from core.orchestrator.events import bus
from core.orchestrator.gates import DEFAULT_GATES, gate_registry
from core.orchestrator.state import AGENT_ORDER, AgentStatus, RunState, RunStatus


REAL_AGENTS: dict[str, type[Agent]] = {
    "ingestion": IngestionAgent,
    "ppg_mapping": PPGMappingAgent,
    "ppg_selection": PPGSelectionAgent,
    "feature_selection": FeatureSelectionAgent,
    "eda": EDAAgent,
    "feature_engineering": FeatureEngineeringAgent,
    "feature_refine": FeatureRefineAgent,
    "modeling": ModelingAgent,
    "results_reasoning": ResultsReasoningAgent,
    "decomposition": DecompositionAgent,
    "simulation": SimulationAgent,
    "optimization": OptimizationAgent,
}


def _build_agent(name: str) -> Agent:
    cls = REAL_AGENTS.get(name)
    return cls() if cls else StubAgent(name=name)


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
                await _build_agent(agent_name).run(run)
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
        run.agents[agent_name].status = prior_status if approved else AgentStatus.failed
        run.status = RunStatus.running if approved else RunStatus.failed
        run.save()
        await bus.publish(
            run.id,
            run.run_dir,
            {"type": "approval_resolved", "agent": agent_name, "approved": approved},
        )
        return approved


async def execute(run: RunState, gates_enabled: bool = True) -> RunState:
    run.status = RunStatus.running
    if gates_enabled:
        run.gates = dict(DEFAULT_GATES)
    else:
        run.gates = {}
    run.save()

    await bus.publish(run.id, run.run_dir, {"type": "run_started", "agents": AGENT_ORDER})

    for agent_name in AGENT_ORDER:
        agent = _build_agent(agent_name)
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

    if run.status != RunStatus.failed:
        run.status = RunStatus.completed
    run.save()
    await bus.publish(run.id, run.run_dir, {"type": "run_finished", "status": run.status.value})
    gate_registry.drop(run.id)
    return run
