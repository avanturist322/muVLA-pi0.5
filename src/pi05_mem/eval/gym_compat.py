"""Gymnasium 0.29 compatibility shim for the simulator wrapper stacks.

Why this file exists
--------------------
mu-VLA runs its simulators under gymnasium 0.29.1; lerobot's pi0.5 stack requires
gymnasium >= 1.0. The two are otherwise compatible in a single process (ManiSkill 3
and SAPIEN 3 install into pi05-mem's venv with no downgrades), and the *only* thing
that breaks is one removed API:

    gymnasium 0.29.1:  gymnasium.Wrapper.__getattr__  ->  forwards to self.env
    gymnasium 1.x:     removed (use env.unwrapped.<attr> or env.get_wrapper_attr)

MIKASA-Robo's wrappers and ManiSkill's `FlattenRGBDObservationWrapper` rely on that
forwarding (e.g. `env.device` read through a stack of wrappers), so building an eval
env under gymnasium 1.3 fails with

    AttributeError: 'StateOnlyTensorToDictWrapper' object has no attribute 'device'

Rather than pin the whole process down to gymnasium 0.29 (which does not support
numpy 2, which torch 2.7 / lerobot need), we restore the forwarding behaviour for the
duration of the eval run. This is deliberately narrow: it only kicks in for attributes
that the wrapper itself does not define.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SENTINEL = "_pi05_mem_getattr_shim"


def _forward_to_inner_env(self, name: str):
    """gymnasium 0.29's Wrapper.__getattr__, minus the deprecation warning."""
    # Guard against recursion while the wrapper is still half-initialised: `self.env`
    # itself goes through __getattr__ if `__init__` has not set it yet.
    if name.startswith("_") or name == "env":
        raise AttributeError(f"accessing private attribute '{name}' is prohibited")
    return getattr(self.env, name)


def patch_wrapper_attribute_forwarding() -> bool:
    """Re-add `Wrapper.__getattr__` if the installed gymnasium dropped it.

    Returns True if the shim was installed, False if it was unnecessary or already
    applied. Idempotent.
    """
    import gymnasium

    wrapper = gymnasium.Wrapper
    if getattr(wrapper, _SENTINEL, False):
        return False
    if "__getattr__" in wrapper.__dict__:
        return False  # gymnasium 0.29 and friends: nothing to do

    wrapper.__getattr__ = _forward_to_inner_env
    setattr(wrapper, _SENTINEL, True)
    logger.info(
        "gymnasium %s: restored Wrapper.__getattr__ forwarding (removed in gymnasium 1.0) "
        "so MIKASA-Robo / ManiSkill wrapper stacks can be built",
        gymnasium.__version__,
    )
    return True


def apply_all() -> None:
    """Apply every shim needed before constructing a simulator env."""
    patch_wrapper_attribute_forwarding()
