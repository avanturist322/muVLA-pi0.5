"""Quick sanity read-out of a run's metrics.jsonl (episode resets, memory norms, loss)."""

import json
import sys
from pathlib import Path


def main(run: str) -> None:
    rows = [json.loads(line) for line in (Path(run) / "metrics.jsonl").read_text().splitlines()]
    n = len(rows)
    resets = sum(r["is_first"] for r in rows)
    print(f"{run}: {n} steps")
    print(f"  episode resets: {resets} total, on {sum(1 for r in rows if r['is_first'])} steps")
    print(f"  -> one reset per {n * 4 / max(1, resets):.1f} stream-steps (episodes are 11-20 long)")
    if "mem_in_norm" in rows[0]:
        norms = [r["mem_in_norm"] for r in rows if "mem_in_norm" in r]
        print(f"  mem_in_norm: min {min(norms):.1f} max {max(norms):.1f} last {norms[-1]:.1f}")
        grads = [r["initial_memory_grad_norm"] for r in rows if "initial_memory_grad_norm" in r]
        print(f"  initial_memory grad: nonzero on {sum(1 for g in grads if g > 0)}/{len(grads)} logged steps")
    losses = [r["loss"] for r in rows]
    print(f"  loss: first10 {sum(losses[:10]) / 10:.4f} -> last10 {sum(losses[-10:]) / 10:.4f}")


if __name__ == "__main__":
    for run in sys.argv[1:]:
        main(run)
