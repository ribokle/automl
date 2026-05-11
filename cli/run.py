"""Typer-based CLI."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.data.ge_runner import capture_baseline
from core.orchestrator.runner import execute
from core.orchestrator.state import AGENT_ORDER, RunState

app = typer.Typer(help="Agentic price & promo optimization")
console = Console()


@app.command()
def run(
    data: Path = typer.Option(Path("data/synthetic.csv"), help="Input CSV path"),
    out: Path = typer.Option(Path("runs"), help="Base directory for run artifacts"),
    no_gates: bool = typer.Option(True, "--no-gates/--with-gates", help="Disable approval gates (default: disabled)"),
) -> None:
    """Execute the full agentic pipeline end-to-end."""
    if not data.exists():
        console.print(f"[red]Data file not found: {data}[/red]")
        raise typer.Exit(code=1)

    state = RunState.new(data_path=str(data.resolve()), run_dir=out / "tmp")
    run_dir = out / state.id
    run_dir.mkdir(parents=True, exist_ok=True)
    state.run_dir = str(run_dir.resolve())
    state.duckdb_path = str((run_dir / "warehouse.duckdb").resolve())
    state.save()

    console.print(f"[cyan]Run {state.id} -> {state.run_dir}[/cyan]")

    asyncio.run(execute(state, gates_enabled=not no_gates))

    table = Table(title=f"Run {state.id}: {state.status.value}")
    table.add_column("Agent")
    table.add_column("Status")
    table.add_column("Confidence")
    table.add_column("Notes")
    for name in AGENT_ORDER:
        ar = state.agents[name]
        table.add_row(name, ar.status.value, f"{ar.confidence:.2f}", ar.reasoning[:80])
    console.print(table)


@app.command()
def baseline_create(
    run_dir: Path = typer.Argument(..., help="Run directory whose warehouse should seed the baseline"),
    name: str = typer.Option("synthetic", help="Baseline name -> core/data/baselines/<name>.json"),
) -> None:
    """Snapshot distribution stats from a clean run to use as a drift baseline."""
    duckdb_path = run_dir / "warehouse.duckdb"
    if not duckdb_path.exists():
        console.print(f"[red]No warehouse.duckdb under {run_dir}[/red]")
        raise typer.Exit(code=1)
    out = Path(__file__).resolve().parents[1] / "core" / "data" / "baselines" / f"{name}.json"
    capture_baseline(duckdb_path, out)
    console.print(f"[green]Wrote baseline -> {out}[/green]")


@app.command()
def seed() -> None:
    """Regenerate the synthetic dataset."""
    from synthetic.generator import write_panel

    repo = Path(__file__).resolve().parents[1]
    write_panel(repo / "data" / "synthetic.csv", repo / "synthetic" / "truth.json")
    console.print("[green]Wrote data/synthetic.csv and synthetic/truth.json[/green]")


if __name__ == "__main__":
    app()
