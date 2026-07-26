"""
rollout.py

The episode loop, shared by both simulators.

Port of mu-VLA's `experiments/ablations/_common/episode_runner.py` and the
receding-horizon branch of its two eval scripts. Two properties are what make eval
match training, and both are checked here rather than assumed:

  1. **Memory resets at the episode boundary.** Every episode is an independent
     trajectory, so the state must start from `initial_memory` - the same thing
     `is_first` does in the episodic dataloader.
  2. **Memory advances once per simulator step.** Training rolled the state forward one
     step at a time, so evaluation must re-query the policy every step and execute only
     the first action of the chunk. `PI05MemPolicy.select_action` enforces
     `n_action_steps == 1`; here we additionally count forwards, memory updates and
     simulator steps and report them, so a silent regression shows up in the log
     instead of as a mysteriously low success rate.

The per-step L1 delta of the memory state is tracked for the same reason mu-VLA tracks
it: a state that never changes means memory is not being written, which looks exactly
like "memory does not help".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .envs import Observation
from .loader import EvalBundle

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    reward: float
    forward_calls: int
    mem_updates: int
    mem_delta_max: float = 0.0
    mem_delta_min: float = float("inf")
    n_action_steps: int = 1
    # Renders taken, and whether this simulator's physics required them. Recorded rather
    # than trusted: the render is load-bearing in MIKASA-Robo, and the failure it guards
    # against is one that leaves every other field looking healthy.
    renders: int = 0
    render_required: bool = False
    error: str | None = None
    video_path: str | None = None

    @property
    def video_written(self) -> bool:
        """The file has to be on disk, not merely named in the record.

        A resumed run rebuilds its earlier episodes from `episodes.jsonl`, so the path
        comes back whether or not the file survived. Counting the name would report a
        full set of rollouts for a video directory that was emptied, and the guard in
        `run_eval` that exists to catch a missing mp4 encoder would never fire.
        """
        return self.video_path is not None and Path(self.video_path).exists()

    @property
    def consistent(self) -> bool:
        """The policy was queried as often as the inference regime says it should be,
        and memory advanced exactly once per query.

        Receding horizon (`n_action_steps == 1`) means one forward per simulator step,
        which is the training schedule. Chunked inference re-queries only when the
        action queue drains, so the expectation is `ceil(steps / n_action_steps)`.
        Counting every *call* instead - which is what this used to do - makes the check
        pass no matter what the policy did, and a silent regression to open-loop
        rollouts would look perfectly consistent.

        An episode that raised makes no claim either way: it was cut off between the
        forward and the step that would have paid for it, so its counts are one apart by
        construction. Reporting that as a regime violation would point at the inference
        code when what happened was a simulator error, which is recorded separately.
        """
        if self.error is not None:
            return True
        expected = -(-self.steps // max(1, self.n_action_steps))  # ceil
        if self.forward_calls != expected:
            return False
        # One render per simulator step, where the simulator's physics needs it. Folded
        # into `consistent` rather than checked separately so that it inherits the guard
        # in `run_eval` that already turns inconsistency into a non-zero exit: a run that
        # stepped the physics without the render is a different experiment, exactly like
        # a run that served cached chunks under a receding-horizon label.
        if self.render_required and self.renders != self.steps:
            return False
        return self.mem_updates in (0, self.forward_calls)

    def as_dict(self) -> dict:
        return {
            "success": int(self.success),
            "steps": self.steps,
            "reward": round(self.reward, 4),
            "forward_calls": self.forward_calls,
            "mem_updates": self.mem_updates,
            "n_action_steps": self.n_action_steps,
            "renders": self.renders,
            "render_required": self.render_required,
            "mem_delta_max": round(self.mem_delta_max, 6),
            "mem_delta_min": (
                None if self.mem_delta_min == float("inf") else round(self.mem_delta_min, 6)
            ),
            "consistent": self.consistent,
            "error": self.error,
            "video": self.video_path,
        }


def build_raw_batch(observation: Observation) -> dict:
    """Assemble the dict the preprocessor expects, matching the training batch exactly.

    Same as `EpisodicLeRobotDataset._build_item` for a batch of one: float CHW images
    in [0, 1] on CPU, float32 state, task as a list of strings. Device placement is the
    preprocessor's `to_device` step, as in training.
    """
    batch: dict = {}
    for key, image in observation.images.items():
        array = np.asarray(image, dtype=np.float32) / 255.0
        batch[key] = torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)
    batch["observation.state"] = torch.from_numpy(
        np.asarray(observation.state, dtype=np.float32)
    ).unsqueeze(0)
    batch["task"] = [observation.task]
    return batch


@torch.no_grad()
def run_episode(
    bundle: EvalBundle,
    adapter,
    *,
    seed: int,
    max_steps: int | None = None,
    video_path: Path | None = None,
) -> EpisodeResult:
    """Roll out one episode with per-step memory.

    The scene is rendered once per simulator step whether or not a video is being
    recorded - see `render_scene`, it is a physics requirement in MIKASA-Robo, and
    making it conditional on recording would give the 5 recorded episodes different
    dynamics from the other 95. The video frame is `[scene | cameras]`: the cameras are
    what the policy actually saw, so the video cannot disagree with the model's input,
    and the scene render is the only view in which the task's objects are visible.
    """
    policy = bundle.policy
    limit = max_steps or adapter.max_steps
    frames: list[np.ndarray] | None = [] if video_path is not None else None

    result = EpisodeResult(
        success=False, steps=0, reward=0.0, forward_calls=0, mem_updates=0,
        n_action_steps=int(getattr(bundle.config, "n_action_steps", 1) or 1),
        # Read off the adapter, not passed in by the caller: the contract belongs to the
        # simulator, and a caller that forgot the flag would silently disable the check.
        render_required=getattr(adapter, "render_scene", None) is not None,
    )

    try:
        # Inside the guard: a failing `adapter.reset` is the most common ManiSkill
        # failure mode, and outside it the exception escaped run_episode and killed the
        # whole 100-episode run before any result was written.
        observation = adapter.reset(seed)
        # Drops the action queue and the memory state; the next forward re-initialises
        # memory from `initial_memory`.
        policy.reset()

        for _ in range(limit):
            batch = bundle.preprocessor(build_raw_batch(observation))

            # `select_action` only runs the network when its action queue is empty;
            # otherwise it pops a cached action. Counting calls rather than forwards
            # would report chunked inference as if it re-queried every step.
            queue = getattr(policy, "_action_queue", None)
            will_requery = queue is None or len(queue) == 0

            mem_ref = policy._mem_state
            mem_before = None if mem_ref is None else mem_ref.detach().clone()

            action = policy.select_action(batch)
            if will_requery:
                result.forward_calls += 1

            mem_after = policy._mem_state
            # Identity, not "is not None": on a popped action the state object is
            # unchanged, and counting that as an update would hide a memory that
            # stopped advancing.
            if mem_after is not None and mem_after is not mem_ref:
                if mem_before is not None:
                    delta = (mem_after.float() - mem_before.float()).abs().mean().item()
                    result.mem_delta_max = max(result.mem_delta_max, delta)
                    result.mem_delta_min = min(result.mem_delta_min, delta)
                # else: first forward of the episode, memory was just initialised from
                # `initial_memory`, so there is no meaningful delta yet.
                result.mem_updates += 1

            action_np = bundle.postprocessor(action).squeeze(0).float().numpy()
            observation, step = adapter.step(action_np)

            # After the step, and unconditionally. After, because mu-VLA found that
            # capturing before it records the post-reset state at frame 0, where
            # Sapien's visual transforms have not synced yet and the cups look sunken
            # for exactly one frame. Unconditionally, because in MIKASA-Robo the render
            # is what keeps the hidden-object trick from corrupting physics, so it must
            # not depend on whether this episode happens to be one of the recorded ones.
            scene = render_scene(adapter)
            if scene is not None:
                result.renders += 1
            if frames is not None:
                frames.append(compose_frame(scene, stack_cameras(observation)))

            result.steps += 1
            result.reward += step.reward
            if step.success:
                result.success = True
                break
            if step.done:
                break
    except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
        logger.exception("episode failed at step %d", result.steps)
        result.error = f"{type(exc).__name__}: {exc}"

    # No terminal frame is appended here any more: capture moved after `adapter.step`,
    # so the frame that shows the success (or the last state before a crash) is already
    # the last one in the list.

    if video_path is not None:
        base = Path(video_path)
        # Both names this episode index could carry. Re-running an env (`--overwrite`,
        # or a resume that dropped a torn tail) can flip the verdict, and every name not
        # written by *this* rollout has to go: otherwise the directory holds
        # `episode_007_success.mp4` beside `episode_007_fail.mp4`, and `ls`, which is how
        # 100 rollouts get triaged, counts one episode twice under contradictory verdicts.
        candidates = [base.with_name(f"{base.stem}_{v}{base.suffix}") for v in ("success", "fail")]
        if frames:
            # A failed episode still gets its video: that is usually the one worth
            # watching. The verdict goes in the filename so 100 rollouts triage by `ls`.
            outcome = "success" if result.success else "fail"
            path = base.with_name(f"{base.stem}_{outcome}{base.suffix}")
            if write_video(frames, path, fps=getattr(adapter, "fps", 10)):
                result.video_path = str(path)
        # Deliberately also when nothing was written - an episode that threw before its
        # first observation, or an encoder that is no longer available. The leftover is
        # then a file whose name asserts a verdict this run did not produce, sitting in
        # a directory that is handed over as the rollout deliverable. Losing a
        # re-generable mp4 is the cheaper of the two failures.
        kept = Path(result.video_path) if result.video_path else None
        for old in candidates:
            if old != kept and old.exists():
                old.unlink()

    return result


class RenderContractError(RuntimeError):
    """A render the simulator's physics depends on did not happen.

    Separate from every other rollout failure because it is the one that used to be
    silent: this function swallowed the exception and returned None, which cost a video
    frame and - after the render became load-bearing - reproduced the ShellGamePush/Pick
    free-fall bug with `all_consistent: true`, `errors: []`, a plausible `mean_steps`
    and a success rate of exactly 0.00. Nothing downstream could tell that apart from
    "the policy cannot do this task".
    """


def render_scene(adapter) -> np.ndarray | None:
    """One scene render, or None for a simulator that offers no scene view.

    Correctness, not decoration - see `MikasaAdapter.render_scene` for why MIKASA-Robo's
    ShellGamePush/Pick physics depends on this call.

    An adapter that *has* a `render_scene` is one whose physics we have contracted to
    step through the renderer, so a failure here is fatal to the episode and is raised.
    An adapter without one (LIBERO) never had the contract and returns None as before.
    Downgrading a failed required render to a warning is what made the original defect
    invisible; the episode counters (`renders` vs `steps`) exist so that even a raise
    that somehow got caught elsewhere still shows up in `result.json`.
    """
    render = getattr(adapter, "render_scene", None)
    if render is None:
        return None
    try:
        return render()
    except Exception as exc:
        raise RenderContractError(
            f"the scene render required once per simulator step failed "
            f"({type(exc).__name__}: {exc}). In MIKASA-Robo this call is what keeps "
            "hide_visual()'s global gpu_apply from committing the render-only cup "
            "teleport into PhysX, so continuing would measure corrupted physics."
        ) from exc


def compose_frame(scene: np.ndarray | None, strip: np.ndarray) -> np.ndarray:
    """`[scene | cameras]` side by side, mirroring mu-VLA's `compose_eval_video`.

    Falls back to the camera strip alone when there is no scene render, so the LIBERO
    adapter and any render failure still produce a watchable video.
    """
    if scene is None:
        return strip
    scene = np.asarray(scene, dtype=np.uint8)
    if scene.ndim != 3 or scene.shape[2] != 3:
        logger.debug("unexpected scene render shape %s, keeping cameras only", scene.shape)
        return strip
    height = max(scene.shape[0], strip.shape[0])
    views = [
        view
        if view.shape[0] == height
        else np.pad(view, ((0, height - view.shape[0]), (0, 0), (0, 0)))
        for view in (scene, strip)
    ]
    return np.concatenate(views, axis=1)


def stack_cameras(observation: Observation) -> np.ndarray:
    """All camera views of one step side by side, as uint8 HWC."""
    views = [np.asarray(image, dtype=np.uint8) for image in observation.images.values()]
    height = max(view.shape[0] for view in views)
    padded = [
        view
        if view.shape[0] == height
        else np.pad(view, ((0, height - view.shape[0]), (0, 0), (0, 0)))
        for view in views
    ]
    return np.concatenate(padded, axis=1)


def write_video(frames: list[np.ndarray], path: Path, *, fps: int = 10) -> bool:
    """Encode a rollout. Never fatal: a missing encoder must not void an eval run.

    Returns whether the file is actually on disk, so the caller can report "0 videos
    written" instead of silently delivering an empty video directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)
    except Exception as exc:  # noqa: BLE001 - the rollout numbers are what matter
        logger.warning("could not write %s (%s: %s)", path, type(exc).__name__, exc)
        return False
    return path.exists() and path.stat().st_size > 0


def summarize(results: list[EpisodeResult]) -> dict:
    """Success rate with a standard error, as mu-VLA's eval scripts report it."""
    successes = np.array([int(r.success) for r in results], dtype=np.float64)
    n = len(successes)
    mean = float(successes.mean()) if n else 0.0
    se = float(successes.std(ddof=0) / np.sqrt(n)) if n else 0.0
    # Every episode that wrote memory at all, including one whose writes were all
    # exactly zero. Filtering those out would hide a partially frozen memory: the
    # reported maximum would come only from the episodes that still worked.
    deltas = [r.mem_delta_max for r in results if r.mem_updates]
    # The floor matters as much as the ceiling: a state that stops moving on some
    # episodes is memory that quietly switched off, and only the minimum shows it.
    floors = [r.mem_delta_min for r in results if r.mem_delta_min != float("inf")]
    return {
        "episodes": n,
        "successes": int(successes.sum()),
        "success_rate": round(mean, 4),
        "success_rate_se": round(se, 4),
        "mean_steps": round(float(np.mean([r.steps for r in results])), 1) if n else 0.0,
        "all_consistent": all(r.consistent for r in results),
        "inconsistent_episodes": [i for i, r in enumerate(results) if not r.consistent],
        "errors": [r.error for r in results if r.error],
        "mem_delta_max": round(max(deltas), 6) if deltas else None,
        "mem_delta_min": round(min(floors), 6) if floors else None,
        "forward_calls": int(sum(r.forward_calls for r in results)),
        "mem_updates": int(sum(r.mem_updates for r in results)),
        # Both totals, so the ratio is checkable from `result.json` alone without
        # re-reading 100 episode records. `render_required` is any() rather than all():
        # the flag comes from the adapter and is therefore constant across an env, so a
        # mixture would mean two adapters served one result file, and treating that as
        # "not required" would switch the guard off in exactly that case.
        "renders": int(sum(r.renders for r in results)),
        "steps_total": int(sum(r.steps for r in results)),
        "render_required": any(r.render_required for r in results),
    }
