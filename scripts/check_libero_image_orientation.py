"""Decide empirically whether LIBERO env frames need the 180-degree rotation.

openvla-oft rotates `agentview_image` by 180 degrees at eval time because its RLDS
training data was stored that way. Whether the LeRobot conversion kept that convention
is not documented anywhere, and getting it wrong feeds the policy upside-down images
that still look plausible to a metric - the success rate just stays at zero.

So measure it. Both the simulator and the dataset show the same table from the same
fixed camera, so the vertical brightness profile (mean over rows) is a stable
signature of orientation. Whichever of {raw, rotated} correlates better with the
dataset's mean profile is the convention the dataset uses.

Run with the eval environment loaded (see scripts/eval_env.sh):
    python scripts/check_libero_image_orientation.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA = Path("data/libero_spatial_image")
TASK_SUITE = "libero_spatial"
NUM_DATASET_FRAMES = 200
NUM_ENV_RESETS = 3


def dataset_profile(column: str = "observation.images.image") -> np.ndarray:
    table = pq.read_table(sorted((DATA / "data").glob("chunk-*/file-*.parquet"))[0], columns=[column])
    blobs = table.column(column).to_pylist()[:NUM_DATASET_FRAMES]
    frames = []
    for blob in blobs:
        raw = blob["bytes"] if isinstance(blob, dict) else blob
        with Image.open(io.BytesIO(raw)) as img:
            frames.append(np.asarray(img.convert("RGB"), dtype=np.float64))
    stack = np.stack(frames)
    return stack.mean(axis=(0, 2, 3))  # mean brightness per row


def env_profile() -> np.ndarray:
    from pi05_mem.eval.envs import LiberoAdapter

    # flip_images=False: we want the raw simulator orientation to compare against.
    adapter = LiberoAdapter(task_suite_name=TASK_SUITE, task_id=0, flip_images=False)
    frames = []
    for seed in range(NUM_ENV_RESETS):
        obs = adapter.reset(seed)
        frames.append(obs.images["observation.images.image"].astype(np.float64))
    adapter.close()
    return np.stack(frames).mean(axis=(0, 2, 3))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


def main() -> None:
    ds = dataset_profile()
    env_raw = env_profile()
    env_rot = env_raw[::-1]  # a 180-degree rotation reverses the row order

    corr_raw = correlation(ds, env_raw)
    corr_rot = correlation(ds, env_rot)
    print(f"row-profile correlation, dataset vs env as-is    : {corr_raw:+.4f}")
    print(f"row-profile correlation, dataset vs env rotated  : {corr_rot:+.4f}")

    if abs(corr_raw - corr_rot) < 0.05:
        print("\nINCONCLUSIVE: the two orientations are too similar to tell apart this way.")
        raise SystemExit(2)
    if corr_rot > corr_raw:
        print("\nVERDICT: rotate env frames 180 degrees -> keep flip_images=True (the default).")
    else:
        print("\nVERDICT: dataset stores the raw simulator orientation -> pass --no-flip-images.")


if __name__ == "__main__":
    main()
