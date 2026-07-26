"""Work out the gripper-action sign convention of a LeRobot LIBERO dataset.

LIBERO/robosuite's OSC controller takes gripper action +1 = close, -1 = open, while
openvla-oft's RLDS pipeline stored [0,1] and needed normalize+invert before stepping
the env. Which convention a LeRobot conversion ended up with is not documented, and
getting it backwards silently destroys the success rate, so measure it: the gripper
finger separation (`observation.state[6] - observation.state[7]`) shrinks while the
gripper is closing, so the action value that precedes shrinking is "close".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/libero_spatial_image")
table = pq.read_table(sorted((root / "data").glob("chunk-*/file-*.parquet"))[0])

action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
frame_index = table.column("frame_index").to_numpy()

grip_action = action[:, 6]
print(f"action[6]: min={grip_action.min():.3f} max={grip_action.max():.3f} "
      f"unique(rounded)={np.unique(np.round(grip_action, 3))[:8]}")

# Finger separation; LIBERO's two gripper joints move symmetrically.
sep = state[:, 6] - state[:, 7]
print(f"state[6]-state[7]: min={sep.min():.4f} max={sep.max():.4f}")

starts = np.flatnonzero(frame_index == 0)
ends = np.append(starts[1:], len(frame_index))

d_sep = np.full(len(sep), np.nan)
for s, e in zip(starts, ends, strict=True):
    d_sep[s : e - 1] = np.diff(sep[s:e])

valid = ~np.isnan(d_sep)
closing = valid & (d_sep < -1e-4)
opening = valid & (d_sep > 1e-4)
print(f"steps: {valid.sum()} usable, {closing.sum()} closing, {opening.sum()} opening")
print(f"mean action[6] while closing: {grip_action[closing].mean():+.3f}")
print(f"mean action[6] while opening: {grip_action[opening].mean():+.3f}")

first_steps = grip_action[starts]
print(f"action[6] at episode start (gripper is open there): "
      f"mean={first_steps.mean():+.3f}")

sign = "+1 = close  ->  matches robosuite directly, pass actions through as-is"
if grip_action[closing].mean() < grip_action[opening].mean():
    sign = "-1 = close  ->  sign must be inverted before env.step"
print(f"\nCONVENTION: {sign}")
