"""
shard.py

One LeRobot v3 dataset root, loaded and indexed: rows, episode boundaries, task
strings and frame sources. `EpisodicLeRobotDataset` streams over one or more of
these (multi-task training draws each episode from a randomly chosen shard).

Two dataset layouts are supported, because the tasks we train on use one each:
MIKASA-Robo (`dtype="video"` cameras, frames pre-decoded into `cache/*.npy`) and
LIBERO (`dtype="image"` cameras with the PNG bytes inline in the data parquet).
See `frame_sources.py`. Only the data files actually present on disk are used, so a
partially downloaded dataset works as long as `meta/` is complete.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .frame_sources import (
    FrameSource,
    NpyCacheFrameSource,
    ParquetImageFrameSource,
    extract_image_blobs,
)

logger = logging.getLogger(__name__)

_DATA_FILE_RE = re.compile(r"chunk-(\d+)/file-(\d+)\.parquet$")


@dataclass(frozen=True)
class Episode:
    """One episode as a half-open `[start, end)` range into the flat row arrays."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class DatasetShard:
    """A single LeRobot v3 dataset root, ready for random episode access."""

    def __init__(self, root: Path | str, *, index: int = 0, name: str | None = None):
        import pyarrow.parquet as pq

        self.root = Path(root)
        self.index = index
        self.name = name or self.root.name

        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps: int = int(info["fps"])
        features = info["features"]
        self.video_camera_keys = [k for k, v in features.items() if v["dtype"] == "video"]
        self.image_camera_keys = [k for k, v in features.items() if v["dtype"] == "image"]
        self.camera_keys: list[str] = self.video_camera_keys + self.image_camera_keys
        if not self.camera_keys:
            raise ValueError(f"{self.root}/meta/info.json declares no image or video features")

        self.camera_shapes: dict[str, tuple[int, int, int]] = {
            cam: tuple(features[cam]["shape"]) for cam in self.camera_keys
        }

        data_files = self._discover_data_files(self.root)
        self._load_rows(pq, data_files, features)
        self._load_tasks(pq)
        self.episodes = self._build_episodes(pq, data_files)
        self._open_frame_sources(data_files)

        self.total_frames = len(self.states)
        self.state_dim = int(self.states.shape[1])
        self.action_dim = int(self.actions.shape[1])

        if not self.episodes:
            raise ValueError(f"{self.root}: no usable episodes found")

        logger.info(
            "shard %s: %d data file(s), %d episodes, %d frames, cameras=%s",
            self.name,
            len(data_files),
            len(self.episodes),
            self.total_frames,
            self.camera_keys,
        )

    # --- loading ----------------------------------------------------------

    @staticmethod
    def _discover_data_files(root: Path) -> list[Path]:
        """Data parquet files present on disk, in (chunk, file) order."""
        files = sorted(
            (root / "data").glob("chunk-*/file-*.parquet"),
            key=lambda p: tuple(int(g) for g in _DATA_FILE_RE.search(str(p)).groups()),
        )
        if not files:
            raise FileNotFoundError(f"no data/chunk-*/file-*.parquet under {root}")
        return files

    def _load_rows(self, pq, data_files: list[Path], features: dict) -> None:
        """Concatenate the per-file columns into flat arrays, in file order."""
        states, actions, task_index, frame_index, episode_index, global_index = [], [], [], [], [], []
        image_blobs: dict[str, list[bytes]] = {k: [] for k in self.image_camera_keys}

        for path in data_files:
            table = pq.read_table(path)
            states.append(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32))
            actions.append(np.asarray(table.column("action").to_pylist(), dtype=np.float32))
            task_index.append(table.column("task_index").to_numpy())
            frame_index.append(table.column("frame_index").to_numpy())
            episode_index.append(table.column("episode_index").to_numpy())
            global_index.append(table.column("index").to_numpy())
            for cam in self.image_camera_keys:
                image_blobs[cam].extend(extract_image_blobs(table.column(cam)))

        self.states = np.concatenate(states)
        self.actions = np.concatenate(actions)
        self.task_index = np.concatenate(task_index)
        self.frame_index = np.concatenate(frame_index)
        self.episode_index = np.concatenate(episode_index)
        # Global row ids, ascending: lets us map meta/episodes' dataset_from/to_index
        # onto local rows even when only some data files were downloaded.
        self._global_index = np.concatenate(global_index)
        self._image_blobs = image_blobs

    def _load_tasks(self, pq) -> None:
        tasks = pq.read_table(self.root / "meta" / "tasks.parquet")
        task_col = "__index_level_0__" if "__index_level_0__" in tasks.column_names else "task"
        names = tasks.column(task_col).to_pylist()
        if "task_index" in tasks.column_names:
            # Do not assume row order matches task_index.
            order = tasks.column("task_index").to_pylist()
            table: dict[int, str] = dict(zip(order, names, strict=True))
            self.tasks = [table[i] for i in sorted(table)]
        else:
            self.tasks = names

    def _build_episodes(self, pq, data_files: list[Path]) -> list[Episode]:
        """Episode boundaries, preferring `meta/episodes` over frame_index heuristics."""
        meta_files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if meta_files:
            episodes = self._episodes_from_meta(pq, meta_files)
            if episodes:
                return episodes
            logger.warning(
                "%s: meta/episodes matched no downloaded data file; falling back to frame_index",
                self.name,
            )
        return self._episodes_from_frame_index()

    def _episodes_from_meta(self, pq, meta_files: list[Path]) -> list[Episode]:
        episodes: list[Episode] = []
        present_min, present_max = int(self._global_index[0]), int(self._global_index[-1])

        for path in meta_files:
            table = pq.read_table(path)
            from_idx = table.column("dataset_from_index").to_numpy()
            to_idx = table.column("dataset_to_index").to_numpy()
            for g_start, g_end in zip(from_idx.tolist(), to_idx.tolist(), strict=True):
                # Skip episodes whose rows were not downloaded.
                if g_start < present_min or g_end - 1 > present_max:
                    continue
                start = int(np.searchsorted(self._global_index, g_start))
                end = int(np.searchsorted(self._global_index, g_end - 1)) + 1
                if self._global_index[start] != g_start or end - start != g_end - g_start:
                    continue  # rows missing in the middle: not a usable episode
                episodes.append(Episode(start, end))
        return episodes

    def _episodes_from_frame_index(self) -> list[Episode]:
        starts = np.flatnonzero(self.frame_index == 0)
        if len(starts) == 0:
            raise ValueError("cannot infer episode boundaries: no row has frame_index == 0")
        ends = np.append(starts[1:], len(self.frame_index))
        return [Episode(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]

    def _open_frame_sources(self, data_files: list[Path]) -> None:
        self.frames: dict[str, FrameSource] = {}

        cache_dir = self.root / "cache"
        for cam in self.video_camera_keys:
            source = NpyCacheFrameSource(cache_dir / f"{cam}.npy")
            if len(source) != len(self.states):
                raise ValueError(
                    f"{self.name}/{cam}: frame cache has {len(source)} frames but "
                    f"{len(self.states)} rows were loaded from {len(data_files)} data file(s). "
                    "The cache written by scripts/predecode_videos.py must cover exactly the "
                    "rows present on disk."
                )
            self.frames[cam] = source

        for cam in self.image_camera_keys:
            h, w, c = self.camera_shapes[cam]
            self.frames[cam] = ParquetImageFrameSource(
                self._image_blobs[cam], expected_shape=(h, w, c)
            )

    # --- item construction ------------------------------------------------

    def num_episodes(self) -> int:
        return len(self.episodes)

    def build_item(self, ep: int, t: int, action_horizon: int) -> dict:
        """One training item: frame `t` of episode `ep`, plus the action chunk."""
        episode = self.episodes[ep]
        length = episode.length
        idx = episode.start + t

        # Clamp past the episode end (repeat the last action), and report which
        # entries were padded so a loss mask can be applied if desired.
        offsets = np.arange(action_horizon)
        raw = t + offsets
        clamped = np.clip(raw, 0, length - 1)
        action = self.actions[episode.start + clamped]
        action_is_pad = raw > (length - 1)

        item = {
            "observation.state": torch.from_numpy(self.states[idx].copy()),
            "action": torch.from_numpy(action.copy()),
            "action_is_pad": torch.from_numpy(action_is_pad.copy()),
            "task": self.tasks[int(self.task_index[idx])],
            "is_first": bool(t == 0),
            "is_last": bool(t == length - 1),
            "episode_index": int(self.episode_index[idx]),
            "frame_index": int(t),
            "dataset_name": self.name,
            "dataset_index": self.index,
        }
        for cam in self.camera_keys:
            frame = np.asarray(self.frames[cam].frame(idx), dtype=np.float32) / 255.0  # HWC [0,1]
            item[cam] = torch.from_numpy(frame).permute(2, 0, 1).contiguous()  # CHW
        return item
