"""Add q01/q99 to a LeRobot v3 dataset's meta/stats.json.

pi0.5 normalizes `observation.state` and `action` with QUANTILES, but not every
published dataset ships q01/q99 - `lerobot/libero_spatial_image` only has
min/max/mean/std. LeRobot's own `augment_dataset_quantile_stats.py` fixes this by
streaming the whole dataset through its video-backed loader; we cannot use it here
(torchcodec is broken in this container, and we deliberately downloaded only part of
the data), so this computes the same quantiles directly from the parquet columns that
are on disk.

Caveat, recorded in the file itself under "q_source": when only some data files were
downloaded, the quantiles come from that subset. That is fine for smoke runs and for
any training that uses the same subset, but stats must be recomputed before training
on the full dataset.

Usage:  python scripts/augment_quantile_stats.py data/libero_spatial_image
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

QUANTILE_KEYS = ("observation.state", "action")


def collect(root: Path, key: str) -> np.ndarray:
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data files under {root}/data")
    chunks = [np.asarray(pq.read_table(f, columns=[key]).column(key).to_pylist(), dtype=np.float64)
              for f in files]
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--force", action="store_true", help="overwrite existing q01/q99")
    args = parser.parse_args()

    stats_path = args.root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())

    changed = []
    for key in QUANTILE_KEYS:
        if key not in stats:
            raise KeyError(f"{key} missing from {stats_path}")
        if "q01" in stats[key] and "q99" in stats[key] and not args.force:
            print(f"{key}: q01/q99 already present, skipping")
            continue
        values = collect(args.root, key)
        q01, q99 = np.quantile(values, 0.01, axis=0), np.quantile(values, 0.99, axis=0)
        # A degenerate dimension (constant across the subset) would make the
        # normalizer divide by ~0; widen it the way lerobot's normalizer expects.
        flat = np.isclose(q99, q01)
        if flat.any():
            print(f"{key}: {int(flat.sum())} constant dim(s), widening by 1e-3")
            q99 = np.where(flat, q01 + 1e-3, q99)
        stats[key]["q01"] = q01.tolist()
        stats[key]["q99"] = q99.tolist()
        # Provenance goes at the top level, not inside the feature entry: every
        # value under a feature is expected to be a numeric array by the loaders.
        stats.setdefault("_quantile_provenance", {})[key] = {
            "frames": int(len(values)),
            "files": [str(p.relative_to(args.root)) for p in
                      sorted((args.root / "data").glob("chunk-*/file-*.parquet"))],
        }
        changed.append(key)
        print(f"{key}: {len(values)} frames, q01={np.round(q01, 4).tolist()}")
        print(f"{' ' * len(key)}  q99={np.round(q99, 4).tolist()}")

    if not changed:
        print("nothing to do")
        return

    backup = stats_path.with_suffix(".json.orig")
    if not backup.exists():
        shutil.copy2(stats_path, backup)
        print(f"original saved to {backup}")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"updated {stats_path}: {', '.join(changed)}")


if __name__ == "__main__":
    main()
