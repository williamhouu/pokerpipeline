#!/usr/bin/env python
"""Refresh the archetype catalog in every saved NLHE prompt file from code.

Generation already stays correct on its own: ``load_preflop_system_prompt``
re-splices the current code catalog into the active prompt on every load (see
``_resync_archetype_catalog``). This script does the same to the saved files on
DISK, so the admin panel's prompt EDITOR also shows the current catalog (and so
``git``-free local prompt copies don't drift visibly). It is idempotent -- run
it any time after editing ``PREFLOP_ARCHETYPE_GUIDANCE``; running it twice is a
no-op.

Targets ``admin_panel/prompts/preflop_system.txt`` and every
``admin_panel/prompts/library/*.txt`` (all NLHE). PLO prompts have no archetype
catalog block, so they are skipped automatically (the resync no-ops without the
``STRATEGIC ARCHETYPES.`` header). Only the catalog block is touched; the
user's voice rules, structure, and examples are left exactly as they are.

    python scripts/resync_prompt_catalogs.py            # rewrite the files
    python scripts/resync_prompt_catalogs.py --check     # report drift, write nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop.explanation_generator import (  # noqa: E402
    _resync_archetype_catalog,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "admin_panel" / "prompts"


def _nlhe_prompt_files() -> list[Path]:
    files: list[Path] = []
    system = PROMPTS_DIR / "preflop_system.txt"
    if system.is_file():
        files.append(system)
    library = PROMPTS_DIR / "library"
    if library.is_dir():
        files.extend(sorted(library.glob("*.txt")))
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="report which files are stale and exit non-zero; write nothing.",
    )
    args = parser.parse_args(argv)

    files = _nlhe_prompt_files()
    if not files:
        print(f"No prompt files found under {PROMPTS_DIR} (nothing to do).")
        return 0

    stale = 0
    for path in files:
        original = path.read_text(encoding="utf-8")
        resynced = _resync_archetype_catalog(original)
        if resynced == original:
            continue
        stale += 1
        if args.check:
            print(f"STALE: {path.relative_to(PROMPTS_DIR.parent.parent)}")
        else:
            path.write_text(resynced, encoding="utf-8")
            print(f"resynced: {path.relative_to(PROMPTS_DIR.parent.parent)}")

    if not stale:
        print(f"All {len(files)} prompt file(s) already current.")
        return 0
    if args.check:
        print(f"\n{stale} file(s) have a stale archetype catalog. "
              "Run without --check to fix.")
        return 1
    print(f"\nResynced {stale} of {len(files)} prompt file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
