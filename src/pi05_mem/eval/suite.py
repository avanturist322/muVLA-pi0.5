"""
suite.py

Evaluate one checkpoint on all 23 MIKASA-Robo tasks and reduce the result to a CSV.

The protocol is mu-VLA's: the same 23 envs, 100
trials, seed 4242424242. Two things are different, both deliberate:

* **envs run in parallel, one subprocess per GPU.** ManiSkill builds a Vulkan context
  per process and dies on it often enough that sharing one process across 23 envs is a
  liability; a subprocess also means a crashed env costs one env, not the run.
* **it resumes.** An env whose `result.json` already exists is skipped, so a job that
  is preempted at env 17 does not redo the first 16.

Usage (from a job script that has sourced scripts/eval_env.sh mikasa):

    python -m pi05_mem.eval.suite \
        --checkpoint runs/config-b-mem/final \
        --data-root data --data shell_game_push_vla_v0,...  \
        --output eval_results/config-b-mem_rh-true \
        --receding-horizon true --num-trials 100
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from .exit_codes import GUARD_EXIT
from .provenance import code_version

logger = logging.getLogger(__name__)

# The 23 MIKASA-Robo tasks of the mu-VLA evaluation protocol, in its order.
# Five of them are the training tasks; the other 18 are held out, which is where the
# memory / no-memory difference is supposed to show.
MIKASA_23 = (
    "ShellGameTouch-VLA-v0",
    "ShellGamePush-VLA-v0",
    "ShellGamePick-VLA-v0",
    "InterceptSlow-VLA-v0",
    "InterceptMedium-VLA-v0",
    "InterceptFast-VLA-v0",
    "InterceptGrabSlow-VLA-v0",
    "InterceptGrabMedium-VLA-v0",
    "InterceptGrabFast-VLA-v0",
    "RotateLenientPos-VLA-v0",
    "RotateLenientPosNeg-VLA-v0",
    "RotateStrictPos-VLA-v0",
    "RotateStrictPosNeg-VLA-v0",
    "TakeItBack-VLA-v0",
    "RememberColor3-VLA-v0",
    "RememberColor5-VLA-v0",
    "RememberColor9-VLA-v0",
    "RememberShape3-VLA-v0",
    "RememberShape5-VLA-v0",
    "RememberShape9-VLA-v0",
    "RememberShapeAndColor3x2-VLA-v0",
    "RememberShapeAndColor3x3-VLA-v0",
    "RememberShapeAndColor5x3-VLA-v0",
)

CSV_FIELDS = (
    "env_id",
    "episodes",
    "successes",
    "success_rate",
    "success_rate_se",
    "mean_steps",
    "all_consistent",
    "mem_delta_max",
    "mem_delta_min",
    "n_action_steps",
    "n_errors",
    "wall_time_s",
    # Provenance. A directory name says which *run* a column belongs to; only this says
    # which weights produced it. Evaluating an intermediate step-NNNNN into the same
    # directory would otherwise be indistinguishable from the final result.
    "checkpoint",
    "status",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on 23 MIKASA tasks")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", required=True, help="dataset dir or comma-separated list")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", default=",".join(MIKASA_23))
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242424242)
    parser.add_argument("--receding-horizon", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--videos-per-env", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--env-timeout",
        type=float,
        default=6 * 3600,
        help="seconds before one env is declared wedged and its GPU reclaimed",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated GPU indices; default is every GPU the job was given",
    )
    parser.add_argument("--overwrite", action="store_true", help="re-run envs that already have a result")
    return parser


def visible_gpus() -> list[str]:
    """Every GPU this job was given.

    Falling back to `["0"]` when the enumeration fails used to look harmless. On the
    8-GPU instance these suites run on it serialises 23 envs onto one device - eight
    times the wall clock with seven eighths of the allocation idle - and says so in a
    single warning line inside a multi-hour log that the job script truncates to the
    last 40 lines. Since three suites is the whole remaining budget, a suite that
    cannot tell how many GPUs it has must stop and let the operator pass `--gpus`.
    """
    explicit = os.environ.get("CUDA_VISIBLE_DEVICES")
    if explicit:
        # Stripped and de-duplicated for the same reason as `--gpus`: an inherited
        # "0, 1" would hand one worker a device id with a leading space.
        return list(dict.fromkeys(g.strip() for g in explicit.split(",") if g.strip()))
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        raise SystemExit(
            f"could not enumerate GPUs ({type(exc).__name__}: {exc}). Refusing to guess: "
            "assuming a single GPU would run the whole suite serially on one device. "
            "Pass --gpus explicitly if that is really what you want."
        ) from exc
    gpus = [line.strip() for line in out.splitlines() if line.strip()]
    if not gpus:
        raise SystemExit("nvidia-smi reported no GPUs")
    return gpus


def env_command(args, env_id: str, out_dir: Path, no_resume: bool = False) -> list[str]:
    command = [
        sys.executable, "-m", "pi05_mem.eval.run_eval",
        "--env", "mikasa",
        "--env-id", env_id,
        "--checkpoint", str(args.checkpoint),
        "--data", args.data,
        "--episodes", str(args.num_trials),
        "--seed", str(args.seed),
        "--dtype", args.dtype,
        "--receding-horizon", args.receding_horizon,
        "--output", str(out_dir / "result.json"),
        "--video-dir", str(out_dir / "videos"),
        "--max-videos", str(args.videos_per_env),
    ]
    if args.data_root is not None:
        command += ["--data-root", str(args.data_root)]
    if args.max_steps is not None:
        command += ["--max-steps", str(args.max_steps)]
    if args.overwrite or no_resume:
        # Without this, --overwrite only bypasses the suite's own cache: run_eval then
        # resumes all 100 episodes out of episodes.jsonl, runs no rollouts, and rewrites
        # the same numbers - now labelled "ok" and timed in seconds, i.e. asserting a
        # fresh measurement. Someone reaching for --overwrite suspects the stored result;
        # handing it back is the one answer that must not happen.
        command += ["--no-resume"]
    return command


def wanted_data(args) -> list[str]:
    """The `data` field a fresh run_eval would write for these arguments.

    Deliberately routed through the same `parse_data_arg` run_eval uses rather than
    re-implementing the comma/`--data-root` expansion here: the two spellings
    `--data a,b` and `--data-root . --data a,b` name the same mixture, and a private
    copy of the rule would drift and start rejecting every cache hit (or, worse,
    accepting the wrong one). Imported inside the function because `train` pulls in
    torch, and this driver otherwise starts in under a second.
    """
    from ..train import parse_data_arg

    return sorted(str(root) for root in parse_data_arg(args.data, args.data_root))


def cached_row(args, env_id: str, result_path: Path) -> dict | None:
    """Reuse a finished env only if its result answers *this* question.

    Resuming into a directory left over from a different checkpoint, or from a 10-trial
    smoke run, would quietly mix incomparable numbers into the final table. Cheap to
    check, and the alternative failure is invisible.
    """
    summary = read_summary(result_path)
    if summary is None:
        return None
    if summary.get("episodes") != args.num_trials:
        logger.warning("[%s] cached result has %s episodes, %d wanted; re-running",
                       env_id, summary.get("episodes"), args.num_trials)
        return None
    # `not in (None, x)` would have grandfathered any result.json written before these
    # fields existed - a missing key is exactly the case where the provenance cannot be
    # checked, so it must re-run rather than be trusted.
    if summary.get("checkpoint") != str(args.checkpoint):
        logger.warning("[%s] cached result is from %s, not %s; re-running",
                       env_id, summary.get("checkpoint"), args.checkpoint)
        return None
    if summary.get("seed") != args.seed:
        logger.warning("[%s] cached result used seed %s, not %s; re-running",
                       env_id, summary.get("seed"), args.seed)
        return None
    # Precision changes the actions; `--max-steps` changes what counts as a failure
    # (a truncated episode is scored as unsuccessful); the dataset mixture sets the
    # normalization quantiles, and pi0.5 discretizes the normalized state into the text
    # prompt. Each of the three makes the cached number an answer to a different
    # question, and none of them is visible in the directory name.
    if summary.get("dtype") != args.dtype:
        logger.warning("[%s] cached result used dtype %s, not %s; re-running",
                       env_id, summary.get("dtype"), args.dtype)
        return None
    if summary.get("max_steps") != args.max_steps:
        logger.warning("[%s] cached result used max_steps %s, not %s; re-running",
                       env_id, summary.get("max_steps"), args.max_steps)
        return None
    if summary.get("data") != wanted_data(args):
        logger.warning("[%s] cached result used a different dataset mixture; re-running",
                       env_id)
        return None
    # The inference regime is the experiment's second axis: the same checkpoint at
    # n_action_steps=1 and n_action_steps=5 are two different measurements, and a
    # directory reused across a `--receding-horizon` flip would silently report one as
    # the other.
    #
    # Under `auto` the regime is decided by the checkpoint, and since the checkpoint has
    # already been matched above, the cached regime has to equal what the resolver would
    # pick. Skipping the check on `auto` - the default - would let an `auto` suite serve
    # a chunked result as a receding-horizon one.
    wanted = {"true": True, "false": False}.get(args.receding_horizon)
    if wanted is None:
        # Imported here rather than at module scope: this driver is a thin subprocess
        # launcher, and `loader` pulls in torch and lerobot. Reimplementing the rule
        # instead is what produced the bug where the fresh and cached paths disagreed.
        from .loader import resolve_inference_regime

        wanted, _ = resolve_inference_regime(
            None, summary.get("memory_meta") or {}, summary.get("n_action_steps") or 1
        )
    if summary.get("receding_horizon") != wanted:
        logger.warning("[%s] cached result has receding_horizon=%s, not %s; re-running",
                       env_id, summary.get("receding_horizon"), wanted)
        return None
    if args.videos_per_env and not summary.get("videos_requested"):
        logger.warning("[%s] cached result was produced without video output and this "
                       "suite wants %d; re-running", env_id, args.videos_per_env)
        return None
    # The last provenance axis, and the one that cost this campaign a re-measurement: the
    # code. A cached result from before the render fix is a different measurement of the
    # same checkpoint - 0.00 against 0.79 on ShellGamePush - and every field above is
    # identical across that boundary, so without this the suite would serve the broken
    # number forever. A missing key means the result predates the field and therefore
    # cannot be vouched for; re-run rather than trust, as with the fields above.
    running = code_version()
    if summary.get("code_version") != running:
        logger.warning("[%s] cached result was produced by code version %s, running %s; "
                       "re-running", env_id, summary.get("code_version"), running)
        return None
    row = _row_from_result(summary)
    # run_eval writes result.json *before* it checks its guards - deliberately, so that
    # a hundred rollouts are not thrown away by a failing assertion - and then exits
    # GUARD_EXIT. So a rejected measurement is a complete, well-formed file on disk, and
    # the second invocation of the suite would serve it as `cached` and score it. The
    # guards have to be re-applied to what was read back, not only to what was just run.
    status = classify(args, env_id, summary, row, "cached")
    if status == "cached":
        logger.info("[%s] already done, skipping", env_id)
    return {"env_id": env_id, "status": status, **row}


def run_one(args, env_id: str, gpu: str) -> dict:
    out_dir = args.output / env_id
    result_path = out_dir / "result.json"
    no_resume = False
    if result_path.exists() and not args.overwrite:
        row = cached_row(args, env_id, result_path)
        if row is not None:
            return row
        rejected = read_summary(result_path) or {}
        recorded = rejected.get("episodes") or 0
        if recorded > args.num_trials:
            # run_eval refuses to shrink a finished result, but only by reading the file
            # the unlink below removes. A spot-check (`eval_suite.sh <cfg> <rh> 5`)
            # pointed at a finished directory would otherwise destroy the 100-episode
            # deliverable it meant to inspect. Returned before that unlink, not after.
            logger.error("[%s] refusing to replace a %d-episode result with a %d-episode "
                         "one; point --output elsewhere, or pass --overwrite to mean it",
                         env_id, recorded, args.num_trials)
            return {"env_id": env_id,
                    "status": f"driver_error(would_shrink {recorded}->{args.num_trials})"}
        # Whether to make run_eval discard episodes.jsonl as well as result.json.
        #
        # Mostly no. `run_meta.json` beside episodes.jsonl records the checkpoint, seed,
        # dtype, max_steps, data mixture and regime, and `resume_episodes` reuses a
        # prefix only when that entire dict matches - so every one of those rejection
        # reasons is already refused at the source, and forcing --no-resume here would
        # throw away good episodes for nothing. Extending a 40-episode prefix to 100 is
        # precisely what resume is for, and it arrives here as a rejected cache.
        #
        # The exception is videos: run_meta does not record whether any were asked for,
        # so a run that wrote none would otherwise resume all 100 episodes, roll out
        # nothing, write no videos, and fail the guard on every future attempt.
        no_resume = bool(args.videos_per_env) and not rejected.get("videos_requested")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    started = time.time()
    logger.info("[%s] starting on GPU %s", env_id, gpu)

    # We have decided to re-run this env, which means whatever result.json is sitting
    # here was either rejected by the guards above or explicitly overwritten. It has to
    # go before the subprocess starts, because if the re-run dies - OOM, Vulkan,
    # preemption - the code below finds the *old* file, cannot tell it apart from a
    # fresh one, and reports the rejected numbers as a `partial` row under the new tag.
    # That is how an rh-true measurement ends up in the rh-false column.
    result_path.unlink(missing_ok=True)

    # SAPIEN compiles PhysX kernels into a cache directory; several processes sharing
    # one cache race and can corrupt it, which shows up as an unrelated env crashing.
    process_env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        SAPIEN_PHYSX_CACHE_ROOT=str(out_dir / ".physx_cache"),
    )
    try:
        with log_path.open("w") as log_file:
            completed = subprocess.run(
                env_command(args, env_id, out_dir, no_resume=no_resume),
                stdout=log_file, stderr=subprocess.STDOUT, env=process_env,
                check=False, timeout=args.env_timeout,
            )
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        # A wedged simulator would otherwise hold its GPU for the whole job and the
        # suite would never finish, so the deadline is the difference between losing
        # one env and losing all of them.
        elapsed = round(time.time() - started, 1)
        logger.error("[%s] TIMED OUT after %.0fs, see %s", env_id, elapsed, log_path)
        # run_eval writes result.json *before* adapter.close(), precisely because SAPIEN
        # teardown can hang. So a timeout can sit on top of a complete set of episodes,
        # and throwing it away costs this env in all three suites at once - the
        # comparison means are complete-case, so one lost env is lost everywhere.
        finished = read_summary(result_path) if result_path.exists() else None
        if finished is not None and finished.get("episodes") == args.num_trials:
            logger.warning("[%s] ...but all %d episodes were already written; the "
                           "timeout was in teardown", env_id, args.num_trials)
            row = _row_from_result(finished)
            # run_eval's guards sit *after* the teardown that just hung, so on this path
            # they never ran at all - and this row is otherwise indistinguishable from a
            # clean one. It was the single loudest way to publish a frozen-memory or
            # mislabelled-regime env as a score.
            return {"env_id": env_id,
                    "status": classify(args, env_id, finished, row, "ok(hung_teardown)"),
                    **row, "wall_time_s": elapsed}
        return {"env_id": env_id, "status": "timeout", "wall_time_s": elapsed}
    elapsed = round(time.time() - started, 1)

    summary = read_summary(result_path) if result_path.exists() else None
    if summary is None:
        logger.error("[%s] FAILED (exit %d) after %.0fs, see %s",
                     env_id, returncode, elapsed, log_path)
        return {"env_id": env_id, "status": f"failed(exit={returncode})",
                "wall_time_s": elapsed}

    row = _row_from_result(summary)
    logger.info("[%s] success_rate=%s (%s/%s) in %.0fs%s",
                env_id, row.get("success_rate"), row.get("successes"), row.get("episodes"),
                elapsed, "  [episode errors]" if row.get("n_errors") else "")
    # run_eval exits 1 when some episodes errored; the numbers are still valid and the
    # error fraction decides. GUARD_EXIT is different in kind: run_eval is saying the
    # file it just wrote is not a measurement of the policy - memory that never moved,
    # an inference regime that does not match the label, a missing video deliverable.
    # Treating that as `partial` would score it, which is what made every one of those
    # guards a no-op from the suite's point of view.
    if returncode == GUARD_EXIT:
        logger.error("[%s] REJECTED by a run_eval guard (exit %d), see %s",
                     env_id, returncode, log_path)
        base = f"guard_failed(exit={returncode})"
    else:
        base = "ok" if returncode == 0 else f"partial(exit={returncode})"
    # `partial` is the other exit that leaves an unchecked result.json: run_eval was
    # killed - OOM, preemption - somewhere after the write, which includes the whole
    # guard block at the end of main. Routing every fresh row through the same
    # classifier as a cached one is also what makes the two agree; they used to
    # disagree about which of a guard failure and an episode-error fraction to name,
    # so a file's recorded reason changed on resume.
    return {"env_id": env_id, "status": classify(args, env_id, summary, row, base),
            **row, "wall_time_s": elapsed}


# An env is allowed a few flaky episodes, but past this share the "success rate" is
# mostly describing the crashes. Same threshold as scripts/build_summary.py, so the
# per-suite summary.csv and the cross-run comparison.csv agree on what counts.
MAX_ERROR_FRACTION = 0.1


def guard_rejection(args, summary: dict) -> str | None:
    """Which run_eval guard this stored summary fails, or None if it passes them all.

    A restatement, over the fields that survive into `result.json`, of the three checks
    `run_eval.main` makes after writing the file. It exists because those checks run
    once, in the process that produced the numbers, while the numbers are read back an
    arbitrary number of times afterwards.
    """
    # `is not True`, not `is False`: run_eval's own guard spells this `not
    # summary["all_consistent"]`, so a file where the key is missing or null fails
    # there and would have passed here. The looser of two restatements of one rule is
    # the one that lets a mislabelled regime through.
    if summary.get("all_consistent") is not True:
        return "regime_mismatch"
    if (summary.get("memory_meta") or {}).get("use_memory") and not summary.get("mem_delta_max"):
        return "memory_frozen"
    # Keyed off what the *stored run* was asked for, not off what this suite wants. A
    # result produced without --video-dir has no videos legitimately; judging it by the
    # current --videos-per-env would stamp it guard_failed forever, and because
    # `cached_row` returns a row rather than None the env would never re-run to clear
    # it. The mismatch between the two requests is a cache-provenance question, and
    # `cached_row` handles it there by re-running.
    if summary.get("videos_requested") and not summary.get("videos_written"):
        return "no_videos"
    return None


def classify(args, env_id: str, summary: dict, row: dict, base: str) -> str:
    """The one verdict for a result.json, whichever path read it back.

    The three call sites used to apply different subsets of these checks: the cached
    path re-applied the guards, the fresh path went by the exit code, and the
    hung-teardown path applied neither. So the same file was given a different status
    depending on which invocation of the suite happened to look at it, and only some of
    those statuses kept it out of the table. Two exits in particular leave a complete,
    plausible, *unchecked* result.json behind - a teardown that hung (run_eval's guards
    live after `adapter.close()`) and a kill after the write - and both were being
    scored.

    Reasons are joined rather than ranked. They are not alternatives: an env whose
    simulator threw on every episode also has memory that never moved, and a status
    naming only the first one sends the reader after the wrong cause.
    """
    reasons = []
    episodes = row.get("episodes") or 0
    if episodes != args.num_trials:
        # A short env is not a comparable success rate, and nothing downstream
        # re-derives the intended trial count from the CSV to notice.
        reasons.append(f"incomplete({episodes}/{args.num_trials})")
    rejection = guard_rejection(args, summary)
    if rejection is not None:
        logger.error("[%s] REJECTED by a guard (%s); not scoring it. Fix the cause and "
                     "re-run this env with --overwrite.", env_id, rejection)
        reasons.append(f"guard_failed({rejection})")
    errors = error_status(row)
    if errors is not None:
        reasons.append(errors)
    return "+".join(reasons) if reasons else base


def error_status(row: dict) -> str | None:
    """Why this row is not a measurement of the policy, or None if it is one.

    Kept as a pure function of the row so that every path that can produce one - a
    fresh run, a cached result.json, a resumed env - classifies identically. When this
    lived inline in `run_one`, a fully-errored env was excluded on the run that
    produced it and averaged in as 0.0 on every later invocation that read it back.
    """
    errors = row.get("n_errors") or 0
    episodes = row.get("episodes") or 0
    if not errors or not episodes:
        return None
    if errors >= episodes:
        # Every episode threw, so the 0.0 success rate measures the crash, not the
        # policy. It must not be averaged in as if it were a score.
        return "all_episodes_errored"
    if errors / episodes > MAX_ERROR_FRACTION:
        return f"mostly_errored({errors}/{episodes})"
    return None


def read_summary(path: Path) -> dict | None:
    """The summary block of a finished env, or None if the file is not usable.

    A job killed mid-write leaves a truncated `result.json`. The resume logic keys off
    existence, so a torn file would be treated as "this env is done" forever and that
    env would silently carry a hole - or worse, a partially parsed number - into the
    final table. Returning None here makes the suite re-run it instead.
    """
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning("%s is unreadable (%s: %s); treating the env as not done",
                       path, type(exc).__name__, exc)
        return None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, dict) or not isinstance(summary.get("episodes"), int):
        logger.warning("%s has no usable summary; treating the env as not done", path)
        return None
    return summary


def _row_from_result(summary: dict) -> dict:
    return {
        "episodes": summary.get("episodes"),
        "successes": summary.get("successes"),
        "success_rate": summary.get("success_rate"),
        "success_rate_se": summary.get("success_rate_se"),
        "mean_steps": summary.get("mean_steps"),
        "all_consistent": summary.get("all_consistent"),
        "mem_delta_max": summary.get("mem_delta_max"),
        "mem_delta_min": summary.get("mem_delta_min"),
        # Carried into the CSV so the inference regime a row was measured under is
        # visible in the table itself, not only in the directory name.
        "n_action_steps": summary.get("n_action_steps"),
        "n_errors": len(summary.get("errors") or []),
        "wall_time_s": summary.get("wall_time_s"),
        "checkpoint": summary.get("checkpoint"),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s suite: %(message)s", force=True
    )
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    envs = [e for e in args.envs.split(",") if e]
    # The empty-string filter matters: `--gpus 0,1,` would otherwise create a worker
    # with CUDA_VISIBLE_DEVICES="", i.e. an env that fails on no visible device.
    #
    # Stripped, not merely tested for content: `--gpus "0, 1"` used to keep the space
    # and set CUDA_VISIBLE_DEVICES=" 1" on one worker. De-duplicated too, because
    # `--gpus 0,0` would put two 3B policies on one device and OOM both.
    gpus = list(dict.fromkeys(g.strip() for g in args.gpus.split(",") if g.strip())) \
        if args.gpus else visible_gpus()
    if not gpus:
        raise SystemExit(f"--gpus {args.gpus!r} names no GPU")
    # `cached_row` reaches `train.parse_data_arg`, which imports torch, and it runs in
    # up to one thread per GPU. Doing that import once here serializes it and turns an
    # ImportError into one clean traceback instead of 23 `driver_error(ImportError)`
    # rows. It saves no memory: threads share this process, so the cost is the same
    # ~1 GB whenever it is paid - only *when* changes.
    #
    # Guarded, because under --overwrite `cached_row` is never called and the import is
    # never needed. This driver is otherwise a thin subprocess launcher that starts in
    # under a second, and making it depend on the training stack unconditionally would
    # mean a broken torch install blocks even the runs that do not touch it.
    if not args.overwrite:
        wanted_data(args)
    logger.info("%d env(s) over %d GPU(s): %s", len(envs), len(gpus), ",".join(gpus))
    (args.output / "suite_config.json").write_text(
        json.dumps({**{k: str(v) for k, v in vars(args).items()}, "gpus": gpus}, indent=2)
    )

    pending: queue.Queue = queue.Queue()
    for env_id in envs:
        pending.put(env_id)
    rows: list[dict] = []
    lock = threading.Lock()
    csv_path = args.output / "summary.csv"

    def worker(gpu: str) -> None:
        while True:
            try:
                env_id = pending.get_nowait()
            except queue.Empty:
                return
            try:
                row = run_one(args, env_id, gpu)
            except Exception as exc:  # noqa: BLE001 - one env must not kill the suite
                logger.exception("[%s] driver error", env_id)
                row = {"env_id": env_id, "status": f"driver_error({type(exc).__name__})"}
            with lock:
                rows.append(row)
                # Rewritten after every env, not once at the end: a suite that runs for
                # hours and is preempted at env 20 should still hand over a readable
                # table for the 19 that finished.
                #
                # Inside its own except, because this call is I/O - ENOSPC, or an NFS
                # stall with eight processes writing for hours - and it sits outside the
                # try that protects `run_one`. Uncaught, it would kill this worker
                # thread through `threading.excepthook` (which does not reach the log
                # the job tails), and its GPU would sit idle for the rest of the suite.
                # The row is already in `rows`, so the final write at the end of `main`
                # recovers the table; a failure here costs one intermediate snapshot.
                try:
                    write_csv(csv_path, rows, envs)
                except Exception:  # noqa: BLE001 - a lost snapshot must not idle a GPU
                    logger.exception("could not refresh %s; continuing", csv_path)

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    started = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    write_csv(csv_path, rows, envs)
    logger.info("wrote %s (%.1f min total)", csv_path, (time.time() - started) / 60)

    scored = [r["env_id"] for r in rows if is_scored(r)]
    missing = [e for e in envs if e not in set(scored)]
    if missing:
        # Loudly, because a suite that returns 0 with 21/23 envs looks finished, and the
        # two holes turn into a wrong mean in the report.
        raise SystemExit(
            f"{len(missing)} of {len(envs)} env(s) have no usable score: "
            + ", ".join(missing)
        )


# Every status prefix `classify` (or the failure paths around it) can produce for a row
# that is not a measurement of the policy. A prefix tuple rather than a set because
# `classify` joins several reasons with "+", and the first reason always comes from
# here. Mirrored by UNUSABLE_PREFIXES in scripts/build_summary.py, which makes the same
# judgement over the CSV this writes.
UNUSABLE_PREFIXES = (
    "incomplete", "guard_failed", "all_episodes_errored", "mostly_errored",
    "failed", "timeout", "driver_error",
)


def is_scored(row: dict) -> bool:
    """Whether this row is a measurement of the policy rather than of a crash.

    A rejected row keeps its numbers - they are worth looking at when working out what
    went wrong - so the success rate alone cannot decide this. The status has to be
    consulted too, otherwise a rejected env is averaged into MEAN like any other.
    """
    if str(row.get("status") or "").startswith(UNUSABLE_PREFIXES):
        return False
    # All three fields, not just the rate: the MEAN row sums `episodes` and `successes`
    # as well, and a result.json carrying a numeric rate beside a null count would raise
    # inside `write_csv`. That call sits *outside* the try/except around `run_one`, so
    # the worker thread would die, its GPU would sit idle for the rest of the suite, and
    # the traceback would go to threading.excepthook rather than to the log the operator
    # is shown at the end of the job.
    return all(
        isinstance(row.get(field), (int, float))
        for field in ("success_rate", "episodes", "successes")
    )


def write_csv(path: Path, rows: list[dict], envs: list[str]) -> None:
    order = {env_id: i for i, env_id in enumerate(envs)}
    ordered = sorted(rows, key=lambda r: order.get(r["env_id"], 1_000))
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)
        scored = [r for r in ordered if is_scored(r)]
        if scored:
            writer.writerow({
                "env_id": "MEAN",
                "episodes": sum(r["episodes"] for r in scored),
                "successes": sum(r["successes"] for r in scored),
                "success_rate": round(sum(r["success_rate"] for r in scored) / len(scored), 4),
                "status": f"{len(scored)}/{len(envs)} envs scored",
            })
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
