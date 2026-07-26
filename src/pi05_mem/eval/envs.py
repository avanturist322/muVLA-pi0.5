"""
envs.py

Simulator adapters for rollout. Each adapter turns a simulator's observation into the
exact keys the policy was trained on, and turns the policy's action back into whatever
the simulator wants.

The mapping is the part that silently ruins evaluations if it is wrong (channel order,
image orientation, gripper sign, state layout), so each adapter states where its
convention comes from - the dataset conversion or mu-VLA's eval scripts.

Both adapters need `apply_gym_compat()` first: the simulator stacks were written for
gymnasium 0.29, which forwarded unknown attributes through Wrapper.__getattr__.
gymnasium 1.x removed that. See gym_compat.py.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pickle import UnpicklingError as pickle_UnpicklingError
from typing import Any

import numpy as np

from .gym_compat import apply_all as apply_gym_compat

logger = logging.getLogger(__name__)

# Both simulator stacks are checkouts, not pip packages, so their locations are
# machine-specific and come from the environment. `scripts/eval_env.sh` sets them:
#   MU_VLA_PATH  - the mu-VLA checkout, for experiments.robot.mikasa_robo.*
#   LIBERO_PATH  - the LIBERO checkout
MU_VLA_PATH = os.environ.get("MU_VLA_PATH", "")
LIBERO_PATH = os.environ.get("LIBERO_PATH", "")


@dataclass
class StepResult:
    success: bool
    done: bool
    reward: float = 0.0


@dataclass
class Observation:
    """What the policy sees: one uint8 HWC image per trained camera key, plus state."""

    images: dict[str, np.ndarray]
    state: np.ndarray
    task: str


# --- MIKASA-Robo ---------------------------------------------------------


@dataclass
class MikasaAdapter:
    """MIKASA-Robo (ManiSkill 3, GPU sim).

    Conventions taken from mu-VLA `experiments/robot/mikasa_robo/`:
      - `obs["rgb"]` is (num_envs, 128, 128, 6): base camera in channels 0:3, wrist in
        3:6 (`mikasa_robo_utils.get_mikasa_images`);
      - `obs["joints"]` is (num_envs, 7) = [x, y, z, roll, pitch, yaw, gripper]
        (`get_mikasa_proprio`);
      - the language instruction arrives in `info["language_instruction"]`;
      - actions are clipped to [-1, 1] and passed through - unlike LIBERO, no gripper
        renormalisation (`run_mikasa_robo_eval.process_action`);
      - `env.step` wants a batched cuda tensor, not a numpy array.
    """

    env_id: str = "RememberColor3-VLA-v0"
    mu_vla_path: str = MU_VLA_PATH
    camera_keys: tuple[str, str] = ("observation.images.top", "observation.images.wrist")
    env: Any = field(default=None, init=False)
    max_steps: int = field(default=0, init=False)
    _task: str = field(default="", init=False)
    _obs: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        import sys

        if not self.mu_vla_path:
            raise ValueError(
                "MikasaAdapter needs the mu-VLA checkout on sys.path: set MU_VLA_PATH "
                "(or pass mu_vla_path=...). See scripts/eval_env.sh."
            )
        if self.mu_vla_path not in sys.path:
            sys.path.insert(0, self.mu_vla_path)
        apply_gym_compat()

        from experiments.robot.mikasa_robo.mikasa_robo_utils import make_eval_env

        self.env, self.max_steps = make_eval_env(self.env_id, num_envs=1)
        logger.info("MIKASA env %s ready, episode_timeout=%d", self.env_id, self.max_steps)

    def render_scene(self):
        """The scene render the rollout loop is required to take once per step.

        `make_eval_env` states the contract explicitly - "the eval loop calls
        `env.render()` itself once per simulator step" (mu-VLA
        `mikasa_robo_utils.py:399`) - and MIKASA-Robo depends on it, not merely for
        video. ShellGamePush/Pick register `goal_site` in `_hidden_objects`, and
        `Actor.hide_visual()` performs a *global* `gpu_apply_rigid_dynamic_data()`
        (mani_skill `utils/structs/actor.py:176-217`), which commits `evaluate()`'s
        render-only `z += HEIGHT_OFFSET = 1000` cup teleport into PhysX. `render()`
        calls `show_visual()` over the same list (`envs/sapien_env.py:1373-1374`) and
        restores a consistent physics state. Skipping it left the cups in free fall
        from 1000 m - vz = -g*dt per step, z ~ -5 m by the end of the episode - which
        is why Push/Pick scored 0.00 with no cup ever visible in the rollout, while
        ShellGameTouch, the one ShellGame env that registers no hidden object, was
        unaffected. Dataset collection never saw it either: it runs num_envs=10, so
        `reconfiguration_freq` is 0 and `goal_site` is hidden once, not per episode.
        """
        from experiments.robot.mikasa_robo.mikasa_robo_utils import render_scene_frame

        return render_scene_frame(self.env)

    def reset(self, seed: int) -> Observation:
        obs, info = self.env.reset(seed=seed)
        from experiments.robot.mikasa_robo.mikasa_robo_utils import get_language_instruction

        self._task = get_language_instruction(self.env, info)
        self._obs = obs
        return self._observation(obs)

    def _observation(self, obs) -> Observation:
        from experiments.robot.mikasa_robo.mikasa_robo_utils import (
            get_mikasa_images,
            get_mikasa_proprio,
        )

        base, wrist = get_mikasa_images(obs)
        return Observation(
            images={self.camera_keys[0]: np.asarray(base), self.camera_keys[1]: np.asarray(wrist)},
            state=np.asarray(get_mikasa_proprio(obs), dtype=np.float32),
            task=self._task,
        )

    def step(self, action: np.ndarray) -> tuple[Observation, StepResult]:
        import torch

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        device = self._obs["rgb"].device if torch.is_tensor(self._obs["rgb"]) else "cuda"
        tensor = torch.from_numpy(action).float().unsqueeze(0).to(device)

        obs, reward, terminated, truncated, info = self.env.step(tensor)
        self._obs = obs

        success = False
        if "success" in info:
            success = bool(np.asarray(_to_numpy(info["success"])).any())
        done = success or bool(
            np.logical_or(np.asarray(_to_numpy(terminated)), np.asarray(_to_numpy(truncated))).any()
        )
        return self._observation(obs), StepResult(
            success=success, done=done, reward=float(np.asarray(_to_numpy(reward)).sum())
        )

    def close(self) -> None:
        if self.env is not None:
            self.env.close()


# --- LIBERO --------------------------------------------------------------


LIBERO_MAX_STEPS = {
    # From mu-VLA's run_libero_eval.py, which took them from openvla-oft.
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# The simulator needs a few steps to settle objects before the policy is asked to act;
# openvla-oft and mu-VLA both use 10 no-op steps.
LIBERO_NUM_STEPS_WAIT = 10
LIBERO_DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


@dataclass
class LiberoAdapter:
    """LIBERO (robosuite / MuJoCo, CPU sim).

    Conventions:
      - images come from `agentview_image` / `robot0_eye_in_hand_image` and are rotated
        180 degrees, matching `libero_utils.get_libero_image`. `flip_images=False`
        disables that if a dataset turns out to store the raw orientation - check with
        `scripts/check_libero_image_orientation.py` rather than guessing;
      - state is [eef_pos(3), eef axis-angle(3), gripper_qpos(2)], the layout
        `run_libero_eval.prepare_observation` builds and the one the LeRobot
        `libero_*_image` datasets store (8 dims);
      - gripper actions are passed through: unlike openvla-oft's RLDS pipeline (which
        needed normalize+invert), the LeRobot conversion already uses robosuite's
        convention, +1 = close. Measured with `scripts/probe_gripper_convention.py`.
    """

    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    resolution: int = 256
    flip_images: bool = True
    libero_path: str = LIBERO_PATH
    camera_keys: tuple[str, str] = ("observation.images.image", "observation.images.wrist_image")
    env: Any = field(default=None, init=False)
    max_steps: int = field(default=0, init=False)
    num_steps_wait: int = field(default=LIBERO_NUM_STEPS_WAIT, init=False)
    _task: str = field(default="", init=False)
    _initial_states: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        import sys

        if not self.libero_path:
            raise ValueError(
                "LiberoAdapter needs the LIBERO checkout on sys.path: set LIBERO_PATH "
                "(or pass libero_path=...). See scripts/eval_env.sh."
            )
        if self.libero_path not in sys.path:
            sys.path.insert(0, self.libero_path)
        apply_gym_compat()

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[self.task_suite_name]()
        if not 0 <= self.task_id < suite.n_tasks:
            raise ValueError(f"task_id {self.task_id} outside 0..{suite.n_tasks - 1}")
        task = suite.get_task(self.task_id)
        self._task = task.language
        self._initial_states = _load_init_states(suite, self.task_id)

        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=self.resolution, camera_widths=self.resolution
        )
        # Seeding matters even with a fixed initial state - it moves object positions.
        self.env.seed(0)
        self.max_steps = LIBERO_MAX_STEPS[self.task_suite_name]
        logger.info(
            "LIBERO env %s task %d (%r) ready, max_steps=%d",
            self.task_suite_name,
            self.task_id,
            self._task,
            self.max_steps,
        )

    def reset(self, seed: int) -> Observation:
        self.env.reset()
        index = seed % len(self._initial_states)
        obs = self.env.set_init_state(self._initial_states[index])
        # Let the scene settle; the policy must not see objects mid-fall.
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION)
        return self._observation(obs)

    def _image(self, obs, key: str) -> np.ndarray:
        img = np.asarray(obs[key])
        if self.flip_images:
            img = img[::-1, ::-1]
        return np.ascontiguousarray(img)

    def _observation(self, obs) -> Observation:
        state = np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)).astype(np.float32),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            )
        )
        return Observation(
            images={
                self.camera_keys[0]: self._image(obs, "agentview_image"),
                self.camera_keys[1]: self._image(obs, "robot0_eye_in_hand_image"),
            },
            state=state,
            task=self._task,
        )

    def step(self, action: np.ndarray) -> tuple[Observation, StepResult]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        obs, reward, done, _info = self.env.step(action)
        # LIBERO reports task completion through `done`.
        return self._observation(obs), StepResult(
            success=bool(done), done=bool(done), reward=float(reward)
        )

    def close(self) -> None:
        if self.env is not None:
            self.env.close()


def _load_init_states(suite, task_id: int):
    """Load a task's initial states, working around torch>=2.6 defaults.

    `LIBERO.benchmark.get_task_init_states` calls `torch.load(path)`; since torch 2.6
    that defaults to `weights_only=True`, which rejects the pickled numpy array LIBERO
    ships. Allow-listing does not help either: the pickle names
    `numpy.core.multiarray._reconstruct`, but under numpy 2 that function reports
    `__module__ == "numpy._core.multiarray"`, so torch's name-keyed allow-list never
    matches.

    So read the file ourselves with `weights_only=False`, but only after checking it
    sits inside LIBERO's own installed data directory. That check is the point: it
    keeps the escape hatch from being reachable with an arbitrary path. Trusting
    LIBERO's data files adds no exposure we do not already have - we import and
    execute LIBERO's Python in this same process.
    """
    import os

    import torch
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    root = os.path.realpath(get_libero_path("init_states"))
    path = os.path.realpath(os.path.join(root, task.problem_folder, task.init_states_file))
    if os.path.commonpath([root, path]) != root:
        raise ValueError(f"init states path {path} escapes {root}")

    try:
        return suite.get_task_init_states(task_id)
    except pickle_UnpicklingError:
        logger.info("loading LIBERO init states directly from %s (torch>=2.6 default)", path)
        return torch.load(path, weights_only=False)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion to axis-angle. Copied from robosuite via mu-VLA."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def make_adapter(kind: str, **kwargs):
    if kind == "mikasa":
        return MikasaAdapter(**kwargs)
    if kind == "libero":
        return LiberoAdapter(**kwargs)
    raise ValueError(f"unknown environment {kind!r}; expected 'mikasa' or 'libero'")
