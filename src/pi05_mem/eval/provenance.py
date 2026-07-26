"""
provenance.py

Which code produced a number.

Its own module, and deliberately free of torch, lerobot and numpy, because three
different callers need it and one of them must stay importable without the ML stack:

  * `run_eval.py` stamps it into `run_meta.json` and into `result.json`'s summary,
  * `suite.py` compares it before serving a cached env (it is a subprocess launcher and
    imports no ML libraries by design - `rebuild_suite_csv.py` loads it on a machine
    where lerobot is not installed at all),
  * `rebuild_suite_csv.py` lists it in `PROVENANCE`, so a table built from two code
    versions is flagged instead of silently pooled.

Why it exists at all: commit `90e24c9` changed `ShellGamePush` from 0.00 to 0.79 on the
same checkpoint, the same 100 seeds, the same dtype, the same `max_steps`, the same
inference regime and the same dataset mixture. Every provenance field the campaign
recorded was identical across that boundary. The resulting `MEAN` row was a number no
single version of the code would produce, and nothing automated could have noticed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# `parents[3]` from `src/pi05_mem/eval/provenance.py` is the repository root. Computed
# from `__file__` rather than from the working directory: the suite fans out worker
# subprocesses whose cwd is not guaranteed, and `git -C .` in the wrong place returns
# either another repository's hash or an error, both of which are worse than "unknown".
REPO_ROOT = Path(__file__).resolve().parents[3]

UNKNOWN = "unknown"


def code_version(root: Path | None = None) -> str:
    """`"<short-hash>"`, `"<short-hash>-dirty"`, or `"unknown"`.

    `-dirty` is not cosmetic. Results measured with uncommitted edits are not
    reproducible from the hash alone, and uncommitted edits are how this bug was found
    in the first place - a run stamped with a clean hash that was not the code that ran
    would be worse than no stamp at all.

    Never raises. A checkout without git, or without a `.git` directory, still has to be
    able to produce results; losing a finished 100-episode run to a provenance lookup
    would be the wrong trade. `"unknown"` is honest, and because it matches no real hash
    it still fails the equality checks that guard cache reuse and table pooling, so the
    safe behaviour is the default rather than something the caller has to remember.
    """
    root = Path(root) if root is not None else REPO_ROOT
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    if not head:
        return UNKNOWN
    try:
        # `--untracked-files=no`: an untracked scratch file does not change
        # what the code does, and marking every run dirty because of one would make the
        # flag useless exactly when it matters.
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        # The hash is known and the cleanliness is not. Reporting the bare hash would
        # assert something unverified, so say so instead.
        return f"{head}-unverified"
    return f"{head}-dirty" if dirty else head
