"""Exit codes shared between `run_eval` (which raises them) and `suite` (which reads
them).

They live in their own module because `suite.py` is a thin subprocess launcher that
deliberately imports neither torch nor lerobot, while `run_eval.py` imports both. The
alternative - the same number written down in two files - is exactly the kind of
duplication that drifts, and here a drift would mean the suite silently scoring a run
that `run_eval` refused.
"""

from __future__ import annotations

# "The file I just wrote is not a measurement of the policy."
#
# Distinct from 1, which `run_eval` also returns when a few episodes errored while the
# numbers stayed usable. The suite decides that case by the error fraction; this one it
# must never score.
GUARD_EXIT = 3

__all__ = ["GUARD_EXIT"]
