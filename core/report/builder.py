"""Report rendering.

`build_html(payload)` materialises the executive report Jinja template into a
self-contained HTML document. `build_pdf(html)` then converts that document
into a PDF via WeasyPrint. Both outputs are byte-identical given the same
payload — used by the insights agent and (eventually) the CLI's
``automl report`` command.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["currency"] = _currency
    env.filters["pct"] = _pct
    env.filters["signed_pct"] = _signed_pct
    env.filters["num"] = _num
    return env


def _currency(v: float | int | None) -> str:
    if v is None or v != v:  # NaN
        return "—"
    n = float(v)
    if abs(n) >= 1_000_000:
        return f"${n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"${n / 1_000:.1f}k"
    return f"${n:,.2f}"


def _pct(v: float | int | None, digits: int = 1) -> str:
    if v is None or v != v:
        return "—"
    return f"{float(v) * 100:.{digits}f}%"


def _signed_pct(v: float | int | None, digits: int = 1) -> str:
    if v is None or v != v:
        return "—"
    n = float(v)
    sign = "+" if n >= 0 else ""
    return f"{sign}{n * 100:.{digits}f}%"


def _num(v: float | int | None, digits: int = 0) -> str:
    if v is None or v != v:
        return "—"
    n = float(v)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.{max(1, digits)}f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.{max(1, digits)}f}k"
    return f"{n:,.{digits}f}"


def build_html(payload: dict[str, Any]) -> str:
    return _env().get_template("report.html.j2").render(**payload)


def build_pdf(html: str) -> bytes:
    from weasyprint import HTML  # heavy import; deferred

    return HTML(string=html).write_pdf()
