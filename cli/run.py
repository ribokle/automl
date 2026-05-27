"""Typer-based CLI."""
from __future__ import annotations

import asyncio
import json
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
    agent_mode: bool = typer.Option(
        True,
        "--agent-mode/--no-agent-mode",
        help="Use LLM-backed agents (default) or force deterministic dry-run fallbacks across every stage.",
    ),
    modelling_grain: str = typer.Option(
        "",
        "--modelling-grain",
        help=(
            "Modelling grain to fit demand at. One of ppg_week (default; current "
            "behaviour), store_ppg_week (Hoch-style, one model per store-PPG), "
            "store_category_week (Hoch 1995 paper grain). Empty means use the "
            "MODELLING_GRAIN env var / global default."
        ),
    ),
) -> None:
    """Execute the full agentic pipeline end-to-end."""
    if not data.exists():
        console.print(f"[red]Data file not found: {data}[/red]")
        raise typer.Exit(code=1)

    from core.config import ModellingGrain

    options: dict = {"agent_mode": agent_mode}
    if modelling_grain:
        try:
            options["modelling_grain"] = ModellingGrain(modelling_grain)
        except ValueError:
            allowed = ", ".join(g.value for g in ModellingGrain)
            console.print(
                f"[red]Invalid --modelling-grain {modelling_grain!r}; expected one of {allowed}[/red]"
            )
            raise typer.Exit(code=1)

    state = RunState.new(
        data_path=str(data.resolve()),
        run_dir=out / "tmp",
        options=options,
    )
    run_dir = out / state.id
    run_dir.mkdir(parents=True, exist_ok=True)
    state.run_dir = str(run_dir.resolve())
    state.duckdb_path = str((run_dir / "warehouse.duckdb").resolve())
    state.save()

    console.print(f"[cyan]Run {state.id} -> {state.run_dir}[/cyan]")

    asyncio.run(execute(state, gates_enabled=not no_gates, agent_mode=agent_mode))

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
def models(
    available_only: bool = typer.Option(
        False, "--available-only", help="Show only models whose dependencies are importable."
    ),
) -> None:
    """List the demand-model library: key, family, problem types, availability."""
    import core.models.library  # noqa: F401 — populate the registry
    from core.models.library import registry

    rows = registry.catalog()
    if available_only:
        rows = [r for r in rows if r["available"]]

    table = Table(title=f"Model library ({len(rows)} models)")
    table.add_column("key")
    table.add_column("family")
    table.add_column("problem types")
    table.add_column("available")
    table.add_column("requires")
    for r in rows:
        table.add_row(
            str(r["key"]),
            str(r["family"]),
            ", ".join(r["problem_types"]),  # type: ignore[arg-type]
            "[green]yes[/green]" if r["available"] else "[yellow]no[/yellow]",
            ", ".join(r["required_packages"]) or "-",  # type: ignore[arg-type]
        )
    console.print(table)


@app.command()
def seed() -> None:
    """Regenerate the synthetic dataset."""
    from synthetic.generator import write_panel

    repo = Path(__file__).resolve().parents[1]
    write_panel(repo / "data" / "synthetic.csv", repo / "synthetic" / "truth.json")
    console.print("[green]Wrote data/synthetic.csv and synthetic/truth.json[/green]")


@app.command("prepare-dominicks")
def prepare_dominicks(
    raw_dir: Path = typer.Option(
        Path("data/dominicks-raw"),
        help="Directory containing the Dominick's category CSVs (any nesting).",
    ),
    out: Path = typer.Option(Path("data/dominicks.csv"), help="Output panel CSV path."),
    categories: str = typer.Option(
        "yogurt,beer",
        help="Comma-separated Dominick's category labels (e.g. yogurt,beer,soft_drinks). "
        "Use 'all' for every known category.",
    ),
    stores: str = typer.Option(
        "",
        help="Optional comma-separated STORE numbers to keep. Empty = all stores.",
    ),
    start_week: int = typer.Option(1, help="Earliest Dominick's WEEK to include."),
    end_week: int = typer.Option(0, help="Latest WEEK to include (0 = no limit)."),
    base_price_window: int = typer.Option(
        13, help="Trailing window (weeks) for non-promo base_price max."
    ),
) -> None:
    """Convert a Dominick's archive into the canonical panel CSV.

    The Kilts data-use agreement forbids redistribution of the raw files —
    download them yourself (https://www.chicagobooth.edu/research/kilts) and
    drop the per-category CSVs anywhere under ``data/dominicks-raw/``.
    """
    from core.data.loaders.dominicks import build_dominicks_panel, coverage_report
    from core.data.schema import REQUIRED_COLUMNS

    cat_list = (
        None
        if categories.strip().lower() in {"all", ""}
        else [c.strip() for c in categories.split(",") if c.strip()]
    )
    store_list = [int(s) for s in stores.split(",") if s.strip()] or None

    panel = build_dominicks_panel(
        raw_dir=raw_dir,
        categories=cat_list,
        stores=store_list,
        start_week=start_week,
        end_week=end_week or None,
        base_price_window=base_price_window,
    )

    missing = [c for c in REQUIRED_COLUMNS if c not in panel.columns]
    if missing:
        console.print(f"[red]Output missing required columns: {missing}[/red]")
        raise typer.Exit(code=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)

    coverage = coverage_report(panel)
    coverage_path = out.parent / f"{out.stem}.coverage.json"
    coverage_path.write_text(json.dumps(coverage, indent=2))

    n_skus = panel["sku"].nunique()
    n_stores = panel["store_id"].nunique()
    n_weeks = panel["week_start"].nunique()
    cats = ", ".join(sorted(panel["category"].dropna().unique().tolist()))
    console.print(
        f"[green]Wrote {len(panel):,} rows -> {out}[/green]\n"
        f"  SKUs: {n_skus} · Stores: {n_stores} · Weeks: {n_weeks}\n"
        f"  Categories: {cats}\n"
        f"  Date range: {panel['week_start'].min()} -> {panel['week_start'].max()}"
    )
    if coverage["constant_columns"] or coverage["all_null_columns"]:
        flat = coverage["constant_columns"] + coverage["all_null_columns"]
        console.print(
            f"[yellow]  Loader-emitted constants: {', '.join(flat)} "
            f"(see {coverage_path.name})[/yellow]"
        )


if __name__ == "__main__":
    app()
