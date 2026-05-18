"""Guard: real source files must not be silently `.gitignore`d.

Background: an unanchored `runs/` pattern in `.gitignore` once swallowed
`web/app/runs/page.tsx` and `web/app/runs/[id]/page.tsx`. The Next.js build
picked them up locally so screenshots worked, but the files were never in
the repo and `main` 404'd on `/runs/<id>` for anyone cloning fresh.

This test fails fast if any source file under the tracked directories is
matched by a `.gitignore` rule, so a stray pattern can't silently take out
a route or a module again.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SOURCE_ROOTS = ("api", "cli", "core", "synthetic", "tests", "web/app", "web/components", "web/lib")
SOURCE_EXTENSIONS = {".py", ".ts", ".tsx", ".json", ".sql", ".yml", ".yaml", ".css"}
SKIP_DIRS = {"__pycache__", "node_modules", ".next", "target", "dbt_packages"}


def _walk_sources() -> list[str]:
    rels: list[str] = []
    for root in SOURCE_ROOTS:
        base = REPO / root
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.suffix not in SOURCE_EXTENSIONS:
                continue
            rels.append(str(p.relative_to(REPO)))
    return rels


def test_gitignore_does_not_swallow_sources() -> None:
    paths = _walk_sources()
    assert paths, "expected to find source files to check"
    # `--no-index` is essential: without it, check-ignore short-circuits on
    # tracked files and would never see a pattern that overreaches into them.
    result = subprocess.run(
        ["git", "check-ignore", "--stdin", "--verbose", "--no-index"],
        input="\n".join(paths),
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    # `git check-ignore --stdin` exits 0 if any path matched (bad), 1 if none
    # matched (good), 128 on usage errors.
    if result.returncode == 0:
        offending = result.stdout.strip()
        raise AssertionError(
            "These source files are silently .gitignore-d — anchor the rule "
            "to the repo root (`/runs/`, not `runs/`) and re-add the files:\n"
            + offending
        )
    assert result.returncode == 1, (
        f"git check-ignore failed unexpectedly (rc={result.returncode}): {result.stderr}"
    )
