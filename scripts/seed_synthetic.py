"""CLI entry to (re)generate the synthetic CPG panel + truth file."""
from __future__ import annotations

import argparse
from pathlib import Path

from synthetic.generator import write_panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/synthetic.csv"))
    p.add_argument("--truth", type=Path, default=Path("synthetic/truth.json"))
    args = p.parse_args()
    write_panel(args.out, args.truth, seed=args.seed)
    print(f"Wrote {args.out} and {args.truth}")


if __name__ == "__main__":
    main()
