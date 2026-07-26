"""
predecode_videos.py

Decode the RememberColor3 videos once into a uint8 array on disk.

Why not torchcodec: the wheel that matches lerobot's pin is built against a newer
torch than our CUDA build, and the container's ffmpeg is 4.4 (libavutil.so.56), so
`torchcodec` cannot load. The ffmpeg *binary* decodes the AV1 streams fine, and the
whole dataset is tiny:

    3858 frames x 2 cameras x 128 x 128 x 3 = 379 MB uint8

so decoding up-front is both simpler and much faster for episodic streaming, which
touches frames in a scattered order.

Layout assumption (verified by `--verify`): each camera has a single mp4 whose frame
count equals `total_frames`, with frames in global dataset order.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "remember_color_3_vla_v0"


def decode_video(path: Path, height: int, width: int, num_frames: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    expected = num_frames * height * width * 3
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if len(proc.stdout) != expected:
        raise RuntimeError(f"{path}: got {len(proc.stdout)} bytes, expected {expected}")
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(num_frames, height, width, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--verify", action="store_true", help="check frame ordering assumptions")
    args = parser.parse_args()

    info = json.loads((args.data / "meta" / "info.json").read_text())
    total_frames = info["total_frames"]
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    table = pq.read_table(args.data / "data" / "chunk-000" / "file-000.parquet")
    cols = table.column_names
    print("parquet columns:", cols)
    print("parquet rows:", table.num_rows)

    if args.verify:
        ep_idx = table.column("episode_index").to_numpy()
        fr_idx = table.column("frame_index").to_numpy()
        idx = table.column("index").to_numpy()
        assert table.num_rows == total_frames, (table.num_rows, total_frames)
        assert np.array_equal(idx, np.arange(total_frames)), "global index is not 0..N-1 in row order"
        assert np.all(np.diff(ep_idx) >= 0), "episodes are not stored in increasing order"
        starts = np.flatnonzero(fr_idx == 0)
        assert len(starts) == info["total_episodes"], (len(starts), info["total_episodes"])
        print(f"verified: {len(starts)} episodes, lengths min/max/mean = "
              f"{np.diff(np.append(starts, total_frames)).min()}/"
              f"{np.diff(np.append(starts, total_frames)).max()}/"
              f"{np.diff(np.append(starts, total_frames)).mean():.1f}")

    cache = args.data / "cache"
    cache.mkdir(exist_ok=True)

    for cam in cameras:
        out = cache / f"{cam}.npy"
        if out.exists():
            print(f"{cam}: already cached ({out.stat().st_size / 1e6:.0f} MB)")
            continue
        h, w = info["features"][cam]["shape"][:2]
        video = args.data / "videos" / cam / "chunk-000" / "file-000.mp4"
        print(f"decoding {video} -> {out}")
        frames = decode_video(video, h, w, total_frames)
        np.save(out, frames)
        print(f"{cam}: {frames.shape} {frames.dtype}, mean={frames.mean():.1f}")

    print("PREDECODE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
