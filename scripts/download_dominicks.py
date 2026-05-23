#!/usr/bin/env python3
"""Download the Dominick's Finer Foods scanner panel from Kilts Center.

Reads a JSON manifest of (filename, url) pairs, downloads each file ONE AT A
TIME so a flaky link doesn't take the whole batch down, tracks the outcome of
every file in a state file, and offers an interactive retry pass for any that
failed.

Stdlib only — no third-party imports — so this runs on a fresh Python 3.11+
install before ``uv sync`` has had a chance to fetch anything.

Typical flow
------------
1. Sign the Kilts data-use agreement and open the download page they give you:
   https://www.chicagobooth.edu/research/kilts/research-data/dominicks
2. Copy the per-file URLs (or just the common URL prefix if all files share
   one) into ``scripts/dominicks_manifest.json`` — start from
   ``scripts/dominicks_manifest.example.json``.
3. Run::

       python scripts/download_dominicks.py

   The script downloads into ``data/dominicks-raw/`` (gitignored), extracts
   any .zip files into per-category subdirs, and writes a state file at
   ``data/dominicks-raw/.download_state.json``.
4. When the first pass finishes, any failures are listed and you'll be
   prompted to retry. If failures look like cert errors you'll also be asked
   whether to retry with TLS verification OFF (corporate-proxy escape hatch).

Re-runs are idempotent: files already on disk are skipped.
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "dominicks_manifest.json"
EXAMPLE_MANIFEST = REPO_ROOT / "scripts" / "dominicks_manifest.example.json"
DEFAULT_DEST = REPO_ROOT / "data" / "dominicks-raw"
STATE_FILENAME = ".download_state.json"
CHUNK = 1024 * 1024  # 1 MiB streaming chunks
USER_AGENT = "automl-dominicks-downloader/1.0"
TIMEOUT_SECS = 120


@dataclass
class Item:
    filename: str
    url: str
    category: Optional[str] = None
    status: str = "pending"  # pending | ok | failed | skipped
    error: Optional[str] = None
    bytes_downloaded: int = 0
    last_attempt: Optional[str] = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


def load_manifest(path: Path) -> list[Item]:
    if not path.exists():
        msg = [f"Manifest not found: {path}"]
        if EXAMPLE_MANIFEST.exists():
            msg.append(f"Copy from: {EXAMPLE_MANIFEST}")
            msg.append("Then fill in `base_url` (or per-file `url`) using URLs from your Kilts download page.")
        sys.exit("\n".join(msg))
    blob = json.loads(path.read_text())
    base = (blob.get("base_url") or "").rstrip("/")
    items: list[Item] = []
    for entry in blob.get("files", []):
        fname = entry["filename"]
        url = entry.get("url")
        if not url:
            if not base:
                sys.exit(
                    f"Entry {fname!r} has no `url` and the manifest has no `base_url`. "
                    f"Either set `base_url` once, or set `url` per file."
                )
            url = f"{base}/{fname}"
        items.append(Item(filename=fname, url=url, category=entry.get("category")))
    if not items:
        sys.exit(f"Manifest {path} contains no files.")
    return items


def save_state(dest: Path, items: list[Item]) -> None:
    state = {
        "updated": utc_now(),
        "files": {item.filename: asdict(item) for item in items},
    }
    (dest / STATE_FILENAME).write_text(json.dumps(state, indent=2))


def build_opener(insecure: bool) -> urllib.request.OpenerDirector:
    ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener


def download_one(
    item: Item,
    dest: Path,
    opener: urllib.request.OpenerDirector,
) -> None:
    target = dest / item.filename
    if target.exists() and target.stat().st_size > 0:
        item.status = "ok"
        item.bytes_downloaded = target.stat().st_size
        print(f"  -> {item.filename}  [skip, exists {human_size(item.bytes_downloaded)}]")
        return
    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        part.unlink()  # restart any half-finished download from scratch

    print(f"  -> {item.filename}  starting...", end="", flush=True)
    with opener.open(item.url, timeout=TIMEOUT_SECS) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        with part.open("wb") as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(
                        f"\r  -> {item.filename}  {pct:3d}%  "
                        f"{human_size(downloaded)}/{human_size(total)}      ",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  -> {item.filename}  {human_size(downloaded)}      ",
                        end="",
                        flush=True,
                    )
    part.rename(target)
    item.bytes_downloaded = downloaded
    item.status = "ok"
    print(f"\r  -> {item.filename}  [ok, {human_size(downloaded)}]            ")


def extract_zip(path: Path, dest: Path, category: Optional[str]) -> None:
    if path.suffix.lower() != ".zip":
        return
    out_dir = dest / (category or path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(out_dir)


def attempt_batch(
    items: list[Item],
    dest: Path,
    opener: urllib.request.OpenerDirector,
    extract: bool,
) -> None:
    for item in items:
        item.last_attempt = utc_now()
        try:
            download_one(item, dest, opener)
            if extract:
                extract_zip(dest / item.filename, dest, item.category)
        except KeyboardInterrupt:
            raise
        except (urllib.error.URLError, urllib.error.HTTPError, ssl.SSLError, OSError, zipfile.BadZipFile) as exc:
            item.status = "failed"
            item.error = f"{type(exc).__name__}: {exc}"
            print(f"\r  -> {item.filename}  [FAIL] {item.error}                ")


def looks_like_cert_error(items: list[Item]) -> bool:
    needles = ("certificate", "cert verify", "ssl", "tls", "self-signed", "unable to get")
    for item in items:
        if not item.error:
            continue
        low = item.error.lower()
        if any(n in low for n in needles):
            return True
    return False


def prompt_yes(question: str, default_yes: bool) -> bool:
    if default_yes:
        return True
    try:
        ans = input(f"{question} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download Dominick's scanner panel from Kilts Center.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help=f"Manifest JSON (default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)})")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"Download dir (default: {DEFAULT_DEST.relative_to(REPO_ROOT)})")
    ap.add_argument("--no-extract", action="store_true",
                    help="Skip auto-extraction of .zip files.")
    ap.add_argument("--insecure", action="store_true",
                    help="Disable TLS verification from the start (corporate-proxy escape hatch).")
    ap.add_argument("--yes", action="store_true",
                    help="Don't prompt — auto-retry failures with the same TLS settings.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only download manifest entries whose filename is in this list.")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    items = load_manifest(args.manifest)
    if args.only:
        wanted = set(args.only)
        items = [i for i in items if i.filename in wanted]
        missing = wanted - {i.filename for i in items}
        if missing:
            print(f"WARNING: --only filenames not in manifest: {sorted(missing)}")
        if not items:
            sys.exit("No items match --only.")

    print(f"Manifest: {args.manifest}  ({len(items)} files)")
    print(f"Dest:     {args.dest}")
    print(f"TLS:      {'DISABLED (insecure)' if args.insecure else 'verified'}\n")

    opener = build_opener(insecure=args.insecure)

    print("=== Pass 1 ===")
    try:
        attempt_batch(items, args.dest, opener, extract=not args.no_extract)
    except KeyboardInterrupt:
        save_state(args.dest, items)
        print("\nInterrupted. State saved.")
        return 130
    save_state(args.dest, items)

    pass_num = 1
    while True:
        failures = [i for i in items if i.status == "failed"]
        if not failures:
            break

        print(f"\n{len(failures)} file(s) failed after pass {pass_num}:")
        for item in failures:
            print(f"  - {item.filename}: {item.error}")

        if not prompt_yes(f"Retry {len(failures)} failed download(s)?", args.yes):
            break

        if not args.insecure and looks_like_cert_error(failures):
            if prompt_yes(
                "Failures look like cert errors. Retry with TLS verification DISABLED?",
                args.yes,
            ):
                opener = build_opener(insecure=True)
                args.insecure = True
                print("WARNING: TLS verification is now DISABLED for the rest of this run.")

        for f in failures:
            f.status = "pending"
            f.error = None

        pass_num += 1
        print(f"\n=== Pass {pass_num} (retrying {len(failures)} file(s)) ===")
        try:
            attempt_batch(failures, args.dest, opener, extract=not args.no_extract)
        except KeyboardInterrupt:
            save_state(args.dest, items)
            print("\nInterrupted. State saved.")
            return 130
        save_state(args.dest, items)

    ok = sum(1 for i in items if i.status == "ok")
    failed = [i for i in items if i.status == "failed"]
    print(f"\n=== Done: {ok}/{len(items)} ok, {len(failed)} failed ===")
    if failed:
        print("Still failing:")
        for f in failed:
            print(f"  - {f.filename}: {f.error}")
        print(f"\nState file: {args.dest / STATE_FILENAME}")
        return 1
    print(f"\nState file: {args.dest / STATE_FILENAME}")
    print("Next: uv run automl prepare-dominicks --categories yogurt,beer --out data/dominicks.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
