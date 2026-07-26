"""Probe whether MIKASA-Robo / LIBERO can actually be constructed and rendered
inside this container. Run with the *simulator's* interpreter, not pi05-mem's:

    MU_VLA_PATH=/path/to/mu-vla python scripts/probe_sim.py mikasa
    LIBERO_PATH=/path/to/LIBERO  python scripts/probe_sim.py libero

Prints one line per checkpoint so a failure is attributable to a stage.
"""

import os
import sys
import traceback
from pathlib import Path

MU_VLA = os.environ.get("MU_VLA_PATH", "")
PI05_MEM = os.environ.get("PI05_MEM_ROOT", str(Path(__file__).resolve().parents[1]))


def _step(label: str, fn):
    try:
        out = fn()
        summary = str(out)
        if len(summary) > 240:  # obs dicts hold whole image tensors
            summary = summary[:240] + " …"
        print(f"OK   {label}: {summary}")
        return out
    except Exception as exc:  # noqa: BLE001 — probe reports every failure mode
        print(f"FAIL {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        sys.exit(1)


def _apply_compat() -> None:
    """Restore the gymnasium-0.29 wrapper behaviour the sim stacks depend on.

    A no-op under mu-VLA's own venvs (gymnasium 0.29 still has __getattr__), needed
    under pi05-mem's venv (gymnasium 1.3).
    """
    sys.path.insert(0, f"{PI05_MEM}/src")
    from pi05_mem.eval import apply_gym_compat

    apply_gym_compat()


def probe_mikasa() -> None:
    sys.path.insert(0, MU_VLA)
    env_id = sys.argv[2] if len(sys.argv) > 2 else "RememberColor3-VLA-v0"
    _apply_compat()

    from experiments.robot.mikasa_robo.mikasa_robo_utils import make_eval_env

    env, timeout = _step(
        "make_eval_env", lambda: make_eval_env(env_id, num_envs=1)
    )
    print(f"     episode_timeout={timeout}")

    obs, info = _step("reset", lambda: env.reset(seed=0))
    _step("obs keys", lambda: sorted(obs.keys()))
    for key, value in obs.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        print(f"     obs[{key}] shape={tuple(shape) if shape else None} dtype={dtype}")

    from experiments.robot.mikasa_robo.mikasa_robo_utils import (
        get_language_instruction,
        render_scene_frame,
    )

    _step("language_instruction", lambda: repr(get_language_instruction(env, info)))
    _step("render", lambda: render_scene_frame(env).shape)

    import torch

    # ManiSkill GPU envs want a batched cuda tensor, exactly as mu-VLA's eval does
    # (run_mikasa_robo_eval.py: torch.from_numpy(action).float().unsqueeze(0).to(dev)).
    device = obs["rgb"].device if torch.is_tensor(obs["rgb"]) else "cuda"
    action = torch.zeros((1, 7), dtype=torch.float32, device=device)
    _step(
        "step(zeros)",
        lambda: [getattr(x, "shape", x) for x in env.step(action)[:4]],
    )
    env.close()


def probe_libero() -> None:
    sys.path.insert(0, MU_VLA)
    _apply_compat()

    from libero.libero import benchmark

    suite = _step(
        "benchmark dict",
        lambda: benchmark.get_benchmark_dict()["libero_spatial"](),
    )
    _step("num_tasks", lambda: suite.n_tasks)

    # mu-VLA's experiments.robot.libero.libero_utils imports tensorflow at module level
    # (only for its RLDS data path), so replicate get_libero_env directly instead.
    import os

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(0)
    print(f"     task={task.name!r} lang={task.language!r}")

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    def build():
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)  # seed affects object positions even with a fixed initial state
        return env, task.language

    env, _ = _step("get_libero_env", build)
    obs = _step("reset", lambda: env.reset())
    _step(
        "obs keys",
        lambda: [k for k in sorted(obs.keys()) if "image" in k or "robot0" in k],
    )
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        if key in obs:
            print(f"     obs[{key}] shape={obs[key].shape} dtype={obs[key].dtype}")
    env.close()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "mikasa"
    if which == "mikasa":
        probe_mikasa()
    elif which == "libero":
        probe_libero()
    else:
        raise SystemExit(f"unknown target {which!r}")
    print("PROBE OK")
