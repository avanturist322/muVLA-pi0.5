"""Evaluation helpers: simulator bootstrap and recurrent-inference episode runner.

Import order matters: `gym_compat` is a plain module with no simulator imports, so it
stays importable (and testable) without ManiSkill or LIBERO installed. The heavier
pieces are imported lazily by name below for the same reason - importing
`pi05_mem.eval` must not drag in a simulator.
"""

from .gym_compat import apply_all as apply_gym_compat

__all__ = [
    "apply_gym_compat",
    "EvalBundle",
    "load_eval_policy",
    "EpisodeResult",
    "build_raw_batch",
    "run_episode",
    "summarize",
    "make_adapter",
    "Observation",
]


def __getattr__(name: str):
    if name in ("EvalBundle", "load_eval_policy"):
        from . import loader

        return getattr(loader, name)
    if name in ("EpisodeResult", "build_raw_batch", "run_episode", "summarize"):
        from . import rollout

        return getattr(rollout, name)
    if name in ("make_adapter", "Observation"):
        from . import envs

        return getattr(envs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
