"""Render a `comparison.csv` into the markdown tables of the experiment report.

The report and the CSV have to carry the same numbers. Copying 23 rows by hand across
three runs is where they stop agreeing, and a transcription slip in a success rate is
invisible to every check downstream - nothing recomputes it, and the CSV that would
contradict it sits in a different file. So the report's results section is generated,
not typed.

Empty cells are rendered as a dash rather than dropped: a task that one run failed to
score is a fact about the experiment, and `build_summary` already refuses to average it
into any column. The `notes` of the MEAN rows carry which tasks those were, and they are
printed verbatim underneath each table.

    python scripts/render_report_results.py \
        --comparison eval_results/comparison.csv \
        --rh-ablation eval_results/comparison_rh_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Every aggregate row `build_summary` writes, in the order it writes them. The tuple is
# not only a label list: the per-task table below decides what to leave out by testing
# `env_id` against it, so an aggregate missing here is not merely rendered unlabelled -
# it is rendered as a task. That is what happened when Table 1's two held-out groups were
# added to `build_summary` and not here: `MEAN_held_out_matched` and
# `MEAN_held_out_novel` appeared among the 23 tasks, each carrying a mean of eleven or
# seven of them, and only the `11/11 envs` in the split column said so.
MEAN_ROWS = (
    "MEAN_train",
    "MEAN_held_out_matched",
    "MEAN_held_out_novel",
    "MEAN_held_out",
    "MEAN_all",
)
# Without the task counts baked in. "train (5)" is a claim about the data rather than a
# reading of it, and the `задач` column beside it already carries the real, complete-case
# count - so when a task is dropped the two disagree in print ("train (5) | 4/5 envs").
#
# The two held-out groups are named by the distinction they exist to draw, not by their
# position in the table: `matched` is a task the policy never trained on whose *kind* of
# remembering the training mixture does contain, `novel` one whose kind it does not. A
# reader who sees "held-out 1 / held-out 2" has to go find out which is which, and the
# whole reason for splitting the block is that the two answer different questions.
MEAN_LABEL = {
    "MEAN_train": "train",
    "MEAN_held_out_matched": "held-out, знакомая семантика памяти",
    "MEAN_held_out_novel": "held-out, новая семантика памяти",
    "MEAN_held_out": "held-out, суммарно",
    "MEAN_all": "все",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a comparison CSV, preserving column order."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def tags_of(fieldnames: list[str]) -> tuple[list[str], str | None]:
    """Recover the run tags and the baseline tag from the header.

    The baseline is the one tag carrying a success rate but no delta column - which is
    exactly how `build_summary` writes it, so nothing has to be passed in twice and the
    rendered table cannot disagree with the file about what it was measured against.
    """
    tags = [f[: -len("__success_rate")] for f in fieldnames if f.endswith("__success_rate")]
    with_delta = {f[: -len("__delta")] for f in fieldnames if f.endswith("__delta")}
    baseline = next((t for t in tags if t not in with_delta), None)
    return tags, baseline


def pct(value: str) -> str:
    """A success rate as a percentage; a blank cell stays visibly blank."""
    try:
        return f"{float(value) * 100:.1f}"
    except (TypeError, ValueError):
        return "—"


def points(value: str) -> str:
    """A delta in percentage points, signed so the direction is unmissable."""
    try:
        return f"{float(value) * 100:+.1f}"
    except (TypeError, ValueError):
        return "—"


def plain(value: str) -> str:
    return value if (value or "").strip() else "—"


def render(rows: list[dict[str, str]], fieldnames: list[str], title: str) -> list[str]:
    """One CSV -> a MEAN table, a per-task table, and any complete-case notes."""
    tags, baseline = tags_of(fieldnames)
    if baseline is None:
        raise SystemExit(f"{title}: no baseline column found; is this a comparison CSV?")
    others = [t for t in tags if t != baseline]

    # The aggregate rows get `delta_sd` / `delta_t`, not the `z` the per-task table
    # carries. `build_summary` blanks `delta_z` on every MEAN row deliberately - the
    # pooled z it used to publish assumed the per-task effects were homogeneous, which on
    # this suite they are not - so this table rendered that column and printed a dash in
    # every significance cell, while the paired t over tasks that replaced it never
    # reached the report at all. The sd is printed beside the t because it is the number
    # that shows why the t is the size it is.
    head = ["строка", "задач"] + [f"{t} %" for t in tags]
    for t in others:
        head += [f"Δ {t} (пп)", "sd (пп)", "t"]

    out = [f"#### {title}", "", f"База: `{baseline}`.", ""]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))

    by_id = {r["env_id"]: r for r in rows}
    for key in MEAN_ROWS:
        row = by_id.get(key)
        if row is None:
            continue
        cells = [MEAN_LABEL.get(key, key), plain(row.get("split", ""))]
        cells += [pct(row.get(f"{t}__success_rate", "")) for t in tags]
        for t in others:
            # The sd goes through `pct`, not `points`: it is in the same percentage
            # points as the delta beside it, but a spread has no direction and the signed
            # form would print "+16.8" as if it did.
            cells += [
                points(row.get(f"{t}__delta", "")),
                pct(row.get(f"{t}__delta_sd", "")),
                plain(row.get(f"{t}__delta_t", "")),
            ]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    # Any task that some run could not score. Printed, never silently dropped.
    notes = [f"- `{r['env_id']}`: {r['notes']}" for r in rows if (r.get("notes") or "").strip()]
    if notes:
        out += ["**Пропуски и оговорки:**", ""] + notes + [""]

    out += ["<details><summary>По задачам</summary>", ""]
    task_head = ["задача", "сплит"] + [f"{t} %" for t in tags]
    for t in others:
        task_head += [f"Δ {t} (пп)", "z"]
    out.append("| " + " | ".join(task_head) + " |")
    out.append("|" + "---|" * len(task_head))
    for row in rows:
        if row["env_id"] in MEAN_ROWS:
            continue
        cells = [f"`{row['env_id']}`", plain(row.get("split", ""))]
        cells += [pct(row.get(f"{t}__success_rate", "")) for t in tags]
        for t in others:
            cells += [points(row.get(f"{t}__delta", "")), plain(row.get(f"{t}__delta_z", ""))]
        out.append("| " + " | ".join(cells) + " |")
    out += ["", "</details>", ""]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render comparison CSVs into markdown")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--rh-ablation", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    blocks: list[str] = []
    for path, title in ((args.comparison, "Эффект памяти"),
                        (args.rh_ablation, "Эффект receding horizon")):
        if path is None:
            continue
        if not path.exists():
            logger.warning("%s does not exist; skipping that table", path)
            continue
        with path.open(newline="") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or []
        blocks += render(read_rows(path), fieldnames, title)

    if not blocks:
        raise SystemExit("nothing to render: no readable comparison CSV")

    text = "\n".join(blocks)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
