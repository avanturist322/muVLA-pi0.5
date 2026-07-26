"""
run_eval.py

Evaluate a pi05-mem checkpoint in MIKASA-Robo or LIBERO.

    PYTHONPATH=$PWD/src .venv/bin/python -m pi05_mem.eval.run_eval \
        --env mikasa --env-id RememberColor3-VLA-v0 \
        --checkpoint runs/verify-tbptt-k2/final \
        --data data/remember_color_3_vla_v0 \
        --episodes 5

    PYTHONPATH=$PWD/src .venv/bin/python -m pi05_mem.eval.run_eval \
        --env libero --task-suite libero_spatial --task-id 0 \
        --checkpoint runs/smoke-libero-mem/final \
        --data data/libero_spatial_image \
        --episodes 5

Memory settings are read from the checkpoint's `memory_meta.json`; they are not CLI
flags, so an eval cannot silently disagree with how the checkpoint was trained.

Both simulators need environment variables set before import (Vulkan/PhysX for
ManiSkill, EGL and LIBERO_CONFIG_PATH for robosuite) - see `scripts/eval_env.sh`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..train import parse_data_arg
from .envs import make_adapter
from .exit_codes import GUARD_EXIT
from .loader import load_eval_policy
from .provenance import code_version
from .rollout import EpisodeResult, run_episode, summarize

logger = logging.getLogger(__name__)

# GUARD_EXIT is imported from `exit_codes`, not restated here. It used to be assigned
# again at this point, which shadowed the import and left the number written down in
# two files - the exact duplication `exit_codes` was split out to remove. A later edit
# to one copy would have made `suite.run_one` compare against a code run_eval no longer
# exits with, quietly turning every guard failure back into a scored `partial` row.


def fail_guard(message: str) -> None:
    """Refuse the run: the file on disk is not a measurement of the policy."""
    logger.error(message)
    raise SystemExit(GUARD_EXIT)


def episode_from_record(record: dict) -> EpisodeResult:
    """Rebuild an already-finished episode from its `episodes.jsonl` line."""
    floor = record.get("mem_delta_min")
    return EpisodeResult(
        success=bool(record["success"]),
        steps=int(record["steps"]),
        reward=float(record["reward"]),
        forward_calls=int(record["forward_calls"]),
        mem_updates=int(record["mem_updates"]),
        mem_delta_max=float(record.get("mem_delta_max") or 0.0),
        mem_delta_min=float("inf") if floor is None else float(floor),
        n_action_steps=int(record.get("n_action_steps", 1)),
        error=record.get("error"),
        video_path=record.get("video"),
    )


def resume_episodes(
    log_path: Path, meta_path: Path, meta: dict, seed: int, limit: int
) -> list[EpisodeResult]:
    """Recover the episodes a preempted run already paid for.

    100 episodes of ManiSkill is roughly half an hour of GPU per env; restarting from
    zero after a preemption at episode 97 throws that away, and across 23 envs and
    three suites it is the difference between a result and a missed deadline.

    Only a prefix that matches this run exactly is reused: same checkpoint, same seed,
    same inference regime, and episode `i` recorded with seed `seed + i`. Anything
    after the first record that fails those checks is dropped, so a half-written line
    or a directory reused for a different checkpoint costs a re-run rather than
    contaminating the numbers.
    """
    if not log_path.exists() or not meta_path.exists():
        return []
    try:
        if json.loads(meta_path.read_text()) != meta:
            logger.warning("%s describes a different run; ignoring it and starting over", log_path)
            return []
    except (OSError, json.JSONDecodeError) as exc:
        # An unreadable meta file used to return silently, and the caller answers an
        # empty prefix by truncating `episodes.jsonl`. That is the one branch here that
        # can destroy finished episodes without saying so: the mismatch branch above at
        # least warns. Keep a copy and name the reason, so a torn meta costs a re-run
        # that is recoverable by hand rather than hours of GPU that are not.
        logger.warning("%s is unreadable (%s); treating the run as fresh. Keeping a copy "
                       "of %s alongside it.", meta_path, exc, log_path)
        backup = log_path.with_suffix(log_path.suffix + ".bak")
        try:
            backup.write_text(log_path.read_text())
        except OSError as copy_failure:
            logger.warning("could not back up %s: %s", log_path, copy_failure)
        return []

    lines = log_path.read_text().splitlines()
    done: list[EpisodeResult] = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
            if record["episode"] != index or record["seed"] != seed + index:
                break
            done.append(episode_from_record(record))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            break

    if len(done) < len(lines):
        # Only when something was actually unusable: rewrite the valid prefix so the
        # next episode is appended after a clean line rather than onto a torn one.
        #
        # Every valid record is kept, including those past `limit`. Trimming to `limit`
        # here would mean that re-running a finished 100-episode env with `--episodes 5`
        # - the obvious way to spot-check a suspicious result - deletes the 95 episodes
        # already paid for, with no backup.
        #
        # The prefix rule is right for a torn tail, which is what a killed job leaves.
        # It is destructive for damage in the middle of the file: one bad line at index
        # 40 of 100 would permanently drop episodes 41-99, half an hour of GPU. Hence
        # the copy - the tail is recoverable by hand if it was ever worth recovering.
        log_path.with_suffix(log_path.suffix + ".bak").write_text(log_path.read_text())
        log_path.write_text("".join(
            json.dumps({"episode": i, "seed": seed + i, **r.as_dict()}) + "\n"
            for i, r in enumerate(done)
        ))
        logger.warning("dropped %d unusable line(s) from the tail of %s",
                       len(lines) - len(done), log_path)

    if done:
        logger.info("resuming after %d episode(s) already recorded in %s", len(done), log_path)
    return done[:limit]


def seed_everything(seed: int) -> None:
    """Seed the policy's RNG, not just the simulator's.

    The env seed alone makes the two checkpoints start from the same initial states,
    which is necessary but not sufficient: pi0.5 samples flow-matching noise from the
    global torch generator on every forward, so without this the comparison between
    config A and config B is unpaired and a resumed job produces different numbers for
    the envs it re-runs. mu-VLA seeds the same way (`set_seed_everywhere`).
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def recorded_episodes(path: Path) -> int | None:
    """How many episodes the result file already on disk covers, if it is readable."""
    try:
        payload = json.loads(path.read_text())
        recorded = payload["summary"]["episodes"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return recorded if isinstance(recorded, int) else None


def write_json_atomically(path: Path, payload: dict) -> None:
    """Write via a temp file and rename.

    A preempted job that dies mid-write would otherwise leave a truncated `result.json`
    that the suite counts as a finished env forever after.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pi05-mem in simulation")
    parser.add_argument("--env", choices=("mikasa", "libero"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        required=True,
        help="dataset dir, or a comma-separated list; must be the SAME mixture the "
        "checkpoint was trained on, because the normalization quantiles come from it",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="prefix for --data entries")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="write one mp4 per episode here (all camera views side by side)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=10,
        help="record only the first N episodes; 0 records none, -1 records all",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="override the env's episode limit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", type=Path, default=None, help="write results json here")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="re-run every episode even if episodes.jsonl already has some of them",
    )
    parser.add_argument(
        "--receding-horizon",
        choices=("auto", "true", "false"),
        default="auto",
        help="auto = on iff the checkpoint uses memory (mu-VLA's rule)",
    )
    # MIKASA
    parser.add_argument("--env-id", default="RememberColor3-VLA-v0")
    # LIBERO
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--no-flip-images",
        action="store_true",
        help="LIBERO: skip the 180-degree rotation (see scripts/check_libero_image_orientation.py)",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    args = build_parser().parse_args()

    receding = {"auto": None, "true": True, "false": False}[args.receding_horizon]
    seed_everything(args.seed)

    # The simulator is built first: it is the thing most likely to fail, and failing
    # before a 3B model is pulled onto the GPU keeps the feedback loop short.
    if args.env == "mikasa":
        adapter = make_adapter("mikasa", env_id=args.env_id)
        run_name = f"{args.env}-{args.env_id}"
    else:
        adapter = make_adapter(
            "libero",
            task_suite_name=args.task_suite,
            task_id=args.task_id,
            flip_images=not args.no_flip_images,
        )
        run_name = f"{args.env}-{args.task_suite}-task{args.task_id}"

    roots = parse_data_arg(args.data, args.data_root)

    bundle = load_eval_policy(
        args.checkpoint,
        roots,
        device=args.device,
        dtype=args.dtype,
        receding_horizon=receding,
    )

    logger.info("evaluating %s: %d episodes, cameras=%s", run_name, args.episodes, bundle.camera_keys)

    start = time.time()
    results: list = []
    # One line per episode, flushed as it happens: a job killed at episode 97 still
    # leaves 96 episodes of evidence, and this is the per-episode statistic the
    # experiment report is built from. It is also what a resumed run picks up from.
    episodes_log = args.output.with_name("episodes.jsonl") if args.output else None
    if episodes_log is not None:
        episodes_log.parent.mkdir(parents=True, exist_ok=True)
        meta_path = episodes_log.with_name("run_meta.json")
        run_meta = {
            "checkpoint": str(args.checkpoint),
            "run": run_name,
            "seed": args.seed,
            "n_action_steps": bundle.config.n_action_steps,
            "receding_horizon": bundle.config.effective_receding_horizon,
            # Precision changes the actions, so resuming a bf16 prefix under fp32 would
            # average two different policies into one success rate.
            "dtype": args.dtype,
            "max_steps": args.max_steps,
            # Same reason: the mixture sets the normalization quantiles, so a prefix
            # recorded under the wrong `--data` is a differently-normalized policy.
            "data": sorted(str(root) for root in roots),
            # And the same reason once more, for the input that changed a success rate by
            # 79 points without changing any of the fields above: the code itself. This
            # makes `resume_episodes` refuse a prefix recorded before a code change, which
            # is the whole point - resuming across the render fix would have spliced
            # broken-physics episodes into a fixed run and passed every guard.
            "code_version": code_version(),
        }
        if args.no_resume:
            episodes_log.write_text("")
        else:
            results = resume_episodes(
                episodes_log, meta_path, run_meta, args.seed, args.episodes)
            if not results:
                episodes_log.write_text("")
        # Atomically, like `result.json`, and for a sharper reason. A torn `run_meta.json`
        # is unparseable, `resume_episodes` treats that as "no usable prefix", and the
        # caller then truncates `episodes.jsonl` - so a job killed inside these few
        # milliseconds destroys every episode already paid for. This file is rewritten at
        # the start of every env, on NFS, with eight processes running for hours.
        write_json_atomically(meta_path, run_meta)

    for episode in range(len(results), args.episodes):
        record = args.video_dir is not None and (
            args.max_videos < 0 or episode < args.max_videos
        )
        # Per episode, not just once at startup, so that episode i is reproducible on
        # its own - a resumed or re-ordered run gives the same numbers.
        seed_everything(args.seed + episode)
        result = run_episode(
            bundle,
            adapter,
            seed=args.seed + episode,
            max_steps=args.max_steps,
            video_path=(args.video_dir / f"episode_{episode:03d}.mp4") if record else None,
        )
        results.append(result)
        if episodes_log is not None:
            with episodes_log.open("a") as handle:
                handle.write(json.dumps(
                    {"episode": episode, "seed": args.seed + episode, **result.as_dict()}
                ) + "\n")
        logger.info(
            "episode %d/%d: success=%s steps=%d forwards=%d mem_updates=%d "
            "mem_delta=[%.5f, %.5f] consistent=%s%s",
            episode + 1,
            args.episodes,
            result.success,
            result.steps,
            result.forward_calls,
            result.mem_updates,
            0.0 if result.mem_delta_min == float("inf") else result.mem_delta_min,
            result.mem_delta_max,
            result.consistent,
            f" ERROR {result.error}" if result.error else "",
        )

    summary = summarize(results)
    summary["run"] = run_name
    summary["checkpoint"] = str(args.checkpoint)
    summary["memory_meta"] = bundle.memory_meta
    summary["seed"] = args.seed
    summary["receding_horizon"] = bundle.config.effective_receding_horizon
    summary["n_action_steps"] = bundle.config.n_action_steps
    # Everything else that decides what the numbers mean, so that `suite.cached_row`
    # can refuse a result produced under different settings. Without these the suite's
    # cache is weaker than run_eval's own episode-level resume, which does check them:
    # a --dtype float32 suite would be served entirely from a bfloat16 cache, and a
    # --max-steps 20 debug pass would poison the full run.
    summary["dtype"] = args.dtype
    summary["max_steps"] = args.max_steps
    # Also in the summary, not only in run_meta.json: the suite caches and the CSVs are
    # built from this file, and `rebuild_suite_csv.PROVENANCE` reads it from here. Two
    # code versions in one MEAN row is the defect this exists to make visible.
    summary["code_version"] = code_version()
    # Which machine, for the failures that are not the code's fault: this campaign lost
    # two eval jobs to `vk::DeviceLostError` on two different nodes, and the surviving
    # results were computed in a different container from the rest of their own suite.
    summary["hostname"] = os.uname().nodename
    # The mixture fixes the pooled normalization quantiles, and pi0.5 discretizes the
    # normalized state into the text prompt - a different mixture is a different input.
    summary["data"] = sorted(str(root) for root in roots)
    summary["videos_written"] = sum(1 for r in results if r.video_written)
    # How many were *asked* for. Without it, `videos_written: 0` is ambiguous between
    # "the video deliverable is missing" and "this run was never asked to write one",
    # and a reader that guesses wrong either poisons a good env with a permanent
    # guard failure or accepts a suite whose videos were silently never produced.
    summary["videos_requested"] = 0 if args.video_dir is None else (
        args.episodes if args.max_videos < 0 else min(args.max_videos, args.episodes)
    )
    summary["wall_time_s"] = round(time.time() - start, 1)
    logger.info("summary: %s", json.dumps(summary, indent=2))

    if args.output:
        episodes = [
            {"episode": i, "seed": args.seed + i, **r.as_dict()} for i, r in enumerate(results)
        ]
        # `resume_episodes` protects episodes.jsonl from a short re-run but not this
        # file, and this is the one the suite caches from and the CSV is built out of.
        # Re-running a finished 100-episode env with `--episodes 5` - the documented way
        # to spot-check a suspicious number - would otherwise replace it with a 5-trial
        # summary sitting in a table headed "100 trials".
        already = recorded_episodes(args.output) if args.output.exists() else None
        if already is not None and already > len(results) and not args.no_resume:
            logger.warning(
                "%s already covers %d episode(s) and this run produced %d; leaving it "
                "in place. Re-run with --no-resume to replace it deliberately.",
                args.output, already, len(results),
            )
        else:
            write_json_atomically(args.output, {"summary": summary, "episodes": episodes})
            logger.info("wrote %s", args.output)

    # Teardown comes *after* the results are on disk. ManiSkill/SAPIEN shutdown raising
    # would otherwise throw away a completed 100-episode run and force a re-run.
    try:
        adapter.close()
    except Exception:  # noqa: BLE001 - a failing teardown must not cost the numbers
        logger.warning("adapter.close() failed after the results were written", exc_info=True)

    # The same number the summary recorded, read back rather than recomputed: two
    # spellings of one rule drift, and here a drift would mean the guard and the field
    # the suite re-checks it from disagree about whether videos were ever asked for.
    wanted_videos = summary["videos_requested"]
    if wanted_videos and not summary["videos_written"]:
        fail_guard(
            f"{wanted_videos} video(s) were requested and none were written - the mp4 "
            f"encoder is unavailable or {args.video_dir} is not writable. The rollouts "
            "are in the result json, but the video deliverable is missing."
        )

    # A memory checkpoint whose memory never moved is the one failure that produces a
    # completely plausible result file: a success rate, all_consistent=True (zero
    # updates is legitimate for the no-memory config), and no error. It would be read as
    # "memory did not help on this task" when what happened is that memory was not
    # running. Nothing downstream can recover this, so it has to fail here.
    #
    # The count of updates cannot detect it. `predict_action_chunk` initialises
    # `_mem_state` lazily on the first forward of an episode, and the rollout counter
    # sees that None -> initial_memory transition as an update, so `mem_updates` is at
    # least one per episode even if every subsequent write is a no-op. The *delta* is
    # what separates a memory that carries state from one pinned to its initial value:
    # it is None when nothing was ever written and exactly 0.0 when the writes changed
    # nothing.
    if bundle.uses_memory and not summary["mem_delta_max"]:
        fail_guard(
            f"{run_name}: the checkpoint declares memory, but over "
            f"{summary['episodes']} episode(s) the memory state never moved "
            f"(mem_delta_max={summary['mem_delta_max']}, "
            f"mem_updates={summary['mem_updates']}). This would be published as a "
            "memory result measured with the memory switched off."
        )

    # The regime the run claims and the regime it executed. `consistent` compares the
    # number of network forwards against ceil(steps / n_action_steps), so a False here
    # means the policy was queried a different number of times than the label implies -
    # e.g. a receding-horizon run that silently served cached chunks. That is a
    # different experiment, not a noisy one, and the summary.csv column recording it is
    # read by nothing downstream, so it has to stop the run.
    if not summary["all_consistent"]:
        fail_guard(
            f"{run_name}: {len(summary['inconsistent_episodes'])} of "
            f"{summary['episodes']} episode(s) queried the policy a different number of "
            f"times than n_action_steps={summary['n_action_steps']} implies "
            f"(episodes {summary['inconsistent_episodes'][:10]}). The run is labelled "
            "with one inference regime and was executed under another."
        )

    # Separate from `all_consistent` even though `consistent` already covers it, because
    # this is the failure whose whole history is "it looked fine". A run that stepped
    # MIKASA-Robo's physics without the per-step render produces the ShellGamePush/Pick
    # free-fall: success rate 0.00, no error, plausible mean_steps. Restating it here
    # gives the reader the two numbers instead of a generic regime-mismatch message.
    if summary["render_required"] and summary["renders"] != summary["steps_total"]:
        fail_guard(
            f"{run_name}: {summary['renders']} scene render(s) over "
            f"{summary['steps_total']} simulator step(s). This simulator's physics is "
            "contracted to be stepped through the renderer once per step; a shortfall "
            "means the hidden-object teleport was committed into PhysX and the objects "
            "are somewhere below the floor. Any success rate from this run is void."
        )

    if summary["errors"]:
        raise SystemExit(f"{len(summary['errors'])} episode(s) failed; see the log")


if __name__ == "__main__":
    main()
