"""
build_summary.py

Merge the per-run `eval_results/<tag>/summary.csv` files into one comparison table.

    python scripts/build_summary.py --results eval_results --out eval_results/comparison.csv

One row per MIKASA task, one column group per run, plus the two columns the experiment
is actually about: the memory-minus-baseline delta and whether it clears the noise.

Success rate over n trials is a binomial proportion, so the standard error of a single
run is sqrt(p(1-p)/n) - about 5 points at p=0.5, n=100. A 3-point difference between
two runs is not a result. The `delta_z` column is the difference divided by the standard
error of the difference, so a reader can see at a glance which rows carry signal.
Rows are also grouped as in Table 1 of arXiv 2606.12497 - the five training tasks, the
eleven held-out tasks whose memory demand the mixture already contains, and the seven
whose demand it does not - because the whole point of the memory mechanism is what
happens on the tasks it never saw, and those two held-out groups ask different questions.

A cell is only filled when the suite actually measured the policy. An env that timed
out, failed to start, or threw on every episode is left blank and named in `notes` -
a crashed env recorded as 0.0 would be indistinguishable from a task the policy simply
cannot do, and would drag the mean down by a fixed amount.

The `MEAN_*` rows are complete-case: a task enters them only if every run scored it, and
the ones dropped are named. Otherwise each column would average over its own surviving
subset, and the difference a reader takes between two column means would partly measure
which run happened to crash where.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# The three groups of Table 1 in arXiv 2606.12497, so the rows here can be read straight
# against the paper's. The split is by *memory semantics*, which is the distinction the
# held-out tasks exist to test: a held-out task whose kind of remembering also occurs in
# the training mixture measures transfer within a demand the policy has seen, while a
# task whose memory demand never occurs there measures whether the mechanism generalises
# to a new one. One mean over all 18 held-out tasks averages those two different claims.

# In the training mixture.
TRAIN_ENVS = {
    "ShellGamePush-VLA-v0",
    "InterceptMedium-VLA-v0",
    "RememberColor5-VLA-v0",
    "TakeItBack-VLA-v0",
    "RememberShapeAndColor3x3-VLA-v0",
}

# Held out, but the kind of remembering is one the training mixture already contains:
# occlusion tracking (ShellGame*), interception timing (Intercept*), colour recall
# (RememberColor*), joint shape-and-colour recall at other cardinalities.
HELD_OUT_MATCHED_ENVS = {
    "ShellGameTouch-VLA-v0",
    "ShellGamePick-VLA-v0",
    "InterceptSlow-VLA-v0",
    "InterceptFast-VLA-v0",
    "InterceptGrabSlow-VLA-v0",
    "InterceptGrabMedium-VLA-v0",
    "InterceptGrabFast-VLA-v0",
    "RememberColor3-VLA-v0",
    "RememberColor9-VLA-v0",
    "RememberShapeAndColor3x2-VLA-v0",
    "RememberShapeAndColor5x3-VLA-v0",
}

# Held out and the memory demand itself is absent from training: shape-only recall (the
# mixture only ever pairs shape with colour) and rotation-angle tracking.
HELD_OUT_NOVEL_ENVS = {
    "RememberShape3-VLA-v0",
    "RememberShape5-VLA-v0",
    "RememberShape9-VLA-v0",
    "RotateLenientPos-VLA-v0",
    "RotateLenientPosNeg-VLA-v0",
    "RotateStrictPos-VLA-v0",
    "RotateStrictPosNeg-VLA-v0",
}

# Checked at import, because the failure is silent otherwise: a task missing from all
# three sets lands in whichever branch `split_of` writes last, or in no group mean at
# all, and the group row still looks like a complete average of its members. 5+11+7=23.
_GROUPS = (TRAIN_ENVS, HELD_OUT_MATCHED_ENVS, HELD_OUT_NOVEL_ENVS)
if sum(len(g) for g in _GROUPS) != 23 or len(set().union(*_GROUPS)) != 23:
    raise SystemExit(
        "the three task groups must partition the 23 MIKASA envs; they currently hold "
        f"{[len(g) for g in _GROUPS]} with {len(set().union(*_GROUPS))} distinct names. "
        "Compare against MIKASA_23 in src/pi05_mem/eval/suite.py."
    )


def split_of(env: str) -> str:
    """Which of Table 1's groups a task belongs to.

    Unknown names return "other" rather than raising: this script is also run over
    synthetic tags in the tests and over scratch directories holding a couple of envs,
    and neither should have to enumerate the benchmark. They are excluded from the group
    means and named in the MEAN_all note, so an unrecognised env cannot hide inside a
    group average.
    """
    if env in TRAIN_ENVS:
        return "train"
    if env in HELD_OUT_MATCHED_ENVS:
        return "held-out-matched"
    if env in HELD_OUT_NOVEL_ENVS:
        return "held-out-novel"
    return "other"

# `guard_failed` is the one that carries a full set of plausible numbers: run_eval
# writes result.json before it checks its guards, so a rejected env has a success rate,
# an episode count and no errors. Reading it as a score is exactly the mistake the
# guard exists to prevent, and it is the only status here that looks fine in the CSV.
# Must stay a superset of `suite.UNUSABLE_PREFIXES`, and matched by prefix rather than
# equality. `suite.classify` joins several reasons with "+", so a fully-crashed env can
# arrive here as `all_episodes_errored+...`. When `all_episodes_errored` and `timeout`
# lived in an exact-equality tuple instead, that was safe only because `error_status`
# happened to be appended last; adding a fourth reason after it would have made a
# crashed env compare unequal, pass `usable()`, and be averaged into MEAN as a real 0.0
# - the one outcome this module exists to prevent. Prefix matching removes the
# dependency on that ordering.
#
# `incomplete` covers an env that produced fewer episodes than the suite asked for: a
# killed or preempted run_eval leaves one behind, and its success rate is a smaller
# sample than every other row in the same column.
UNUSABLE_PREFIXES = ("failed", "driver_error", "guard_failed", "mostly_errored",
                     "incomplete", "all_episodes_errored", "timeout")
# Above this share of crashed episodes the surviving trials are not a fair estimate.
MAX_ERROR_FRACTION = 0.1
# Below this, the standard error of the mean per-task delta is float residue rather than a
# measurement, and the t it would produce is meaningless. The deltas arrive as differences
# of four-decimal success rates, so any real spread is at least ~1e-4; 1e-9 sits five
# orders of magnitude below that and well above double-precision noise.
SD_FLOOR = 1e-9


def read_run(path: Path) -> dict[str, dict]:
    with path.open() as handle:
        return {row["env_id"]: row for row in csv.DictReader(handle)}


def as_float(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def usable(entry: dict) -> str | None:
    """None if the row is a real measurement, else why it is not.

    `suite.py` already refuses to average these into its own MEAN. The same judgement
    has to be made here, because this is the table a human reads.
    """
    if not entry:
        return "missing"
    status = (entry.get("status") or "").strip()
    if status.startswith(UNUSABLE_PREFIXES):
        return status
    if as_float(entry.get("success_rate")) is None:
        return status or "no score"
    # The inference regime the row was measured under. `all_consistent=False` means the
    # policy was queried a different number of times than `n_action_steps` implies, so
    # the row belongs to a different experiment than its column heading claims - the
    # receding-horizon column would be reporting chunked inference. run_eval's guard
    # normally stops this at the source; the column is re-read here because this table
    # is also built from result directories produced before that guard existed.
    #
    # Spelled "not true" rather than "is false", to match `suite.guard_rejection`. The
    # equality form passed a missing or blank cell, which is exactly the case the reason
    # above describes: an older result directory may not carry the column at all, so for
    # its stated purpose the check was a no-op. Unverifiable is not the same as fine.
    if str(entry.get("all_consistent") or "").strip().lower() != "true":
        return "inference regime unverified"
    errors = as_float(entry.get("n_errors")) or 0.0
    # Required, not defaulted: a row with a rate and no episode count would otherwise
    # pass, enter a MEAN row, move the mean, and contribute nothing to the variance -
    # making the aggregate z read as if that task carried no sampling error at all.
    episodes = as_float(entry.get("episodes"))
    if episodes is None or episodes <= 0:
        return "no episode count"
    if errors / episodes > MAX_ERROR_FRACTION:
        return f"{int(errors)}/{int(episodes)} episodes errored"
    return None


def se(rate: float | None, episodes: float | None) -> float | None:
    """Standard error of a success rate, so a delta can be read against its own noise.

    Agresti-Coull rather than the textbook sqrt(p(1-p)/n): the plain form collapses to
    exactly 0 at p=0 and p=1, which would leave `delta_z` blank on precisely the rows
    where the effect is largest - a held-out task the baseline never solves and the
    memory policy always does. Two pseudo-successes and two pseudo-failures keep the
    error finite at the boundary and shift it by under a point anywhere else
    (p=0.5, n=100: 0.0490 against 0.0500).
    """
    if rate is None or not episodes:
        return None
    adjusted = (rate * episodes + 2.0) / (episodes + 4.0)
    return math.sqrt(max(adjusted * (1.0 - adjusted), 0.0) / (episodes + 4.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge eval runs into one CSV")
    parser.add_argument("--results", type=Path, default=Path("eval_results"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--baseline", default="config-a-nomem_rh-false",
        help="run tag the deltas are measured against",
    )
    args = parser.parse_args()

    runs = {}
    for summary in sorted(args.results.glob("*/summary.csv")):
        # A leading underscore marks a scratch directory (smoke runs, reruns). Merging
        # a 2-episode smoke in as a fourth run column would look like a fourth result.
        if summary.parent.name.startswith("_"):
            continue
        runs[summary.parent.name] = read_run(summary)
    if not runs:
        raise SystemExit(f"no <tag>/summary.csv under {args.results}")

    tags = sorted(runs)
    # The delta columns are the point of this table. When the baseline tag was absent
    # the deltas were simply not emitted, and the result was a CSV of three success-rate
    # columns that looks complete - a typo in `--baseline`, or a baseline suite that
    # died before writing its summary.csv, both produce it. Passing `--baseline ''` is
    # the way to ask for a table without deltas on purpose.
    if args.baseline and args.baseline not in runs:
        raise SystemExit(
            f"baseline run {args.baseline!r} has no summary.csv under {args.results}; "
            f"found: {', '.join(tags)}. Every delta column would be missing and the "
            "table would not say so. Pass --baseline '' to build one without deltas."
        )
    base = runs.get(args.baseline)
    envs = [
        e for e in dict.fromkeys(env for run in runs.values() for env in run)
        if not e.startswith("MEAN")
    ]

    # Every column has to be the same size sample. Nothing upstream enforces this:
    # `suite.classify` compares an env's episode count against *that suite's own*
    # --num-trials, so a whole suite run at the wrong N is `status=ok` on all 23 rows.
    # `python -m pi05_mem.eval.suite --run config-b-mem --num-trials 20` into a fresh
    # directory is one
    # keystroke away from doing it, and the resulting column merges in looking complete:
    # the per-task z stays honest (its SE uses n=20) so nothing screams, and a 20-trial
    # column is averaged into MEAN beside 100-trial ones. Only `__episodes` would tell.
    sizes = {}
    for tag in tags:
        seen = {int(n) for env in envs
                if usable(runs[tag].get(env, {})) is None
                and (n := as_float(runs[tag].get(env, {}).get("episodes"))) is not None}
        if seen:
            sizes[tag] = seen
    distinct = sorted(set().union(*sizes.values())) if sizes else []
    if len(distinct) > 1:
        raise SystemExit(
            "these runs were not measured at the same number of trials: "
            + "; ".join(f"{tag}={sorted(n)}" for tag, n in sorted(sizes.items()))
            + ". Averaging them into one MEAN row would compare samples of different "
            "sizes. Re-run the short one at the full trial count, or point --results at "
            "a tree containing only the runs you mean to compare."
        )

    fields = ["env_id", "split"]
    for tag in tags:
        fields += [f"{tag}__success_rate", f"{tag}__successes", f"{tag}__episodes"]
    if base is not None:
        for tag in tags:
            if tag != args.baseline:
                # `delta_z` is per-task only; `delta_sd`/`delta_t` are MEAN-row only. See
                # the MEAN block below for why the aggregate cannot be a z.
                fields += [
                    f"{tag}__delta", f"{tag}__delta_z",
                    f"{tag}__delta_sd", f"{tag}__delta_t",
                ]
    fields.append("notes")

    rows = []
    for env in envs:
        row = {"env_id": env, "split": split_of(env)}
        notes = []
        rates: dict[str, float | None] = {}
        for tag in tags:
            entry = runs[tag].get(env, {})
            problem = usable(entry)
            if problem:
                notes.append(f"{tag}: {problem}")
                rates[tag] = None
                row[f"{tag}__success_rate"] = ""
                row[f"{tag}__successes"] = ""
                row[f"{tag}__episodes"] = entry.get("episodes", "")
                continue
            rates[tag] = as_float(entry.get("success_rate"))
            row[f"{tag}__success_rate"] = entry.get("success_rate", "")
            row[f"{tag}__successes"] = entry.get("successes", "")
            row[f"{tag}__episodes"] = entry.get("episodes", "")
        # Per-env standard errors, kept on the row so the MEAN rows can build their own
        # z out of them. Not a CSV field: `extrasaction="ignore"` drops it on write.
        row["_se"] = {}
        if base is not None:
            base_rate = rates.get(args.baseline)
            base_se = se(base_rate, as_float(base.get(env, {}).get("episodes")))
            row["_se"][args.baseline] = base_se
            for tag in tags:
                if tag == args.baseline:
                    continue
                rate = rates.get(tag)
                run_se = se(rate, as_float(runs[tag].get(env, {}).get("episodes")))
                row["_se"][tag] = run_se
                if rate is None or base_rate is None:
                    row[f"{tag}__delta"] = row[f"{tag}__delta_z"] = ""
                    continue
                delta = rate - base_rate
                row[f"{tag}__delta"] = round(delta, 4)
                pooled = math.hypot(base_se or 0.0, run_se or 0.0)
                row[f"{tag}__delta_z"] = round(delta / pooled, 2) if pooled else ""
        row["notes"] = "; ".join(notes)
        # Whether every run measured this task. Only these envs enter the means, so all
        # columns describe the same set of tasks - see the complete-case note below.
        row["_complete"] = all(rates.get(tag) is not None for tag in tags)
        rows.append(row)

    # Split means, then the overall mean, so the held-out number is not diluted by the
    # five tasks the policy was trained on. `env_rows` is a snapshot: averaging over the
    # live list would fold the first two mean rows into the third.
    env_rows = list(rows)
    for label, subset in (
        ("MEAN_train", [r for r in env_rows if r["split"] == "train"]),
        # The two halves of Table 1's held-out block, then their union. The union row is
        # kept because it is the number the earlier campaign reported, so dropping it
        # would make the two write-ups look like they measured different suites.
        ("MEAN_held_out_matched",
         [r for r in env_rows if r["split"] == "held-out-matched"]),
        ("MEAN_held_out_novel",
         [r for r in env_rows if r["split"] == "held-out-novel"]),
        ("MEAN_held_out",
         [r for r in env_rows if r["split"].startswith("held-out")]),
        ("MEAN_all", env_rows),
    ):
        # Complete-case: an env enters the mean only if *every* run scored it. Averaging
        # each column over "whatever is non-empty in that column" silently compares
        # different task sets - if a crash costs the memory run one hard task, its mean
        # rises against a baseline mean that still carries that task, and the difference
        # a reader computes between the two columns is an artefact of the crash.
        contributing = [r for r in subset if r["_complete"]]
        dropped = [r["env_id"] for r in subset if not r["_complete"]]
        mean_row = {"env_id": label, "split": f"{len(contributing)}/{len(subset)} envs"}
        for field in fields[2:-1]:
            # A z-score is a difference divided by its own standard error, and the mean
            # of several of them is not the z of their mean - it is not a z at all, and
            # it reads far too small: averaging a +4.0 and a +0.2 gives 2.1, as if the
            # aggregate effect were weaker than the strongest task, when pooling
            # independent tasks makes the aggregate *more* significant, not less.
            # Recomputed properly below.
            #
            # `__delta_sd` and `__delta_t` are skipped for the opposite reason: they are
            # aggregate-only quantities, blank on every per-task row, so averaging them
            # would be averaging an empty column. Both are set explicitly below.
            if field.endswith(("__delta_z", "__delta_sd", "__delta_t")):
                continue
            values = [as_float(r.get(field)) for r in contributing]
            values = [v for v in values if v is not None]
            mean_row[field] = round(sum(values) / len(values), 4) if values else ""
        for tag in tags:
            if base is None or tag == args.baseline:
                continue
            # The aggregate test treats the *task* as the sampling unit, not the episode.
            #
            # This used to divide the mean delta by the quadrature sum of the per-task
            # binomial standard errors over n - a fixed-effect pooling, valid only if the
            # per-task deltas are homogeneous, i.e. if every task shares one true effect
            # and differs only by sampling noise. On this suite they emphatically do not:
            # the memory deltas run from -0.43 (TakeItBack) to +0.09, and two of 23 tasks
            # carry the whole aggregate. Under that heterogeneity the within-episode SE is
            # the wrong denominator by a large factor, and the old code turned a mean
            # delta of -3.65 pp into "z = -4.32" - a decisive-looking result - where the
            # paired test over tasks gives t = -1.69 on 22 df, which is not significant.
            # The comment that used to sit here ("pooling independent tasks makes the
            # aggregate *more* significant, not less") is true only in the homogeneous
            # case it silently assumed.
            #
            # So: paired t over the per-task deltas, which is what generalising from "these
            # 23 tasks" to "MIKASA-Robo tasks" actually requires. The sd is reported beside
            # it because it is the number that shows *why* the t is small.
            deltas = [
                d for r in contributing
                if (d := as_float(r.get(f"{tag}__delta"))) is not None
            ]
            n = len(deltas)
            mean_delta = sum(deltas) / n if n else None
            if n > 1 and mean_delta is not None:
                # Sample sd, ddof=1: these 23 tasks are a sample of tasks, not the
                # population, which is the entire premise of testing over them.
                sd = math.sqrt(sum((d - mean_delta) ** 2 for d in deltas) / (n - 1))
                mean_row[f"{tag}__delta_sd"] = round(sd, 4)
                stderr = sd / math.sqrt(n)
                # `> SD_FLOOR`, not `if stderr`. Deltas that are equal in the data are not
                # bitwise equal after a round-trip through the CSV's four decimals, so the
                # sd of six identical -0.10 deltas comes out at ~1e-17 rather than 0, and
                # the bare truthiness check let that through as t = -1.6e16 - a number that
                # reads as overwhelming significance and is pure float residue. Caught by
                # `test_zero_between_task_variance_yields_no_t`.
                mean_row[f"{tag}__delta_t"] = (
                    round(mean_delta / stderr, 2) if stderr > SD_FLOOR else ""
                )
            else:
                # One task is not a sample of tasks. Emitting a t here would be a number
                # with no df behind it, which is worse than an empty cell.
                mean_row[f"{tag}__delta_sd"] = ""
                mean_row[f"{tag}__delta_t"] = ""
            # Deliberately blank, not recomputed: the only aggregate this table used to
            # publish was the invalid one, and leaving a plausible number in a column
            # named `delta_z` next to the correct `delta_t` guarantees someone quotes the
            # wrong one. The per-task rows keep their `delta_z`, where it is a real z.
            mean_row[f"{tag}__delta_z"] = ""
        if dropped:
            mean_row["notes"] = (
                f"complete-case over {len(contributing)}/{len(subset)} envs; "
                f"excluded everywhere: {', '.join(dropped)}"
            )
        rows.append(mean_row)

    out = args.out or args.results / "comparison.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{out}  ({len(envs)} envs x {len(tags)} runs: {', '.join(tags)})")


if __name__ == "__main__":
    main()
