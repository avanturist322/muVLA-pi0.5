"""
episodic_dataset.py

Sequential (episodic) streaming over one or more LeRobot v3 datasets, ported from
mu-VLA's `prismatic/vla/datasets/mikasa_episodic_dataset.py`.

Recurrent memory needs the batch to be *time-coherent*: slot `i` of consecutive
batches must be consecutive steps of the same episode, so that `mem_state[i]` carried
from one training step to the next belongs to the same trajectory.

The dataset therefore keeps `batch_size` independent infinite streams. Each stream
picks a dataset (multi-task) and then an episode inside it, walks that episode step by
step, and jumps to another episode. Items are yielded round-robin - stream 0, stream 1,
..., stream B-1, stream 0, ... - so a `DataLoader(batch_size=B, shuffle=False,
num_workers=0)` reassembles exactly one item per stream per batch, with `batch[i]`
always coming from stream `i`.

`is_first` marks the first step of an episode (memory must be reset there) and
`is_last` the final one. Crossing a dataset boundary is just an ordinary episode
change, so it is also an `is_first` step: memory never leaks between tasks.

Multi-task parity with mu-VLA: the dataset for the next episode is drawn *per episode,
per stream*, uniformly over the configured datasets by default (mu-VLA:
`rng.choice(self.env_names)`), regardless of how many episodes each one holds. Pass
`dataset_weights` for a different mixture.

Per-root loading lives in `shard.py`; combined normalization statistics in
`dataset_stats.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .dataset_stats import load_dataset_stats  # noqa: F401  (re-exported)
from .shard import DatasetShard

logger = logging.getLogger(__name__)


def normalize_roots(roots: Path | str | Sequence[Path | str]) -> tuple[Path, ...]:
    """Accept a single dataset root or a sequence of them."""
    if isinstance(roots, (str, Path)):
        return (Path(roots),)
    out = tuple(Path(r) for r in roots)
    if not out:
        raise ValueError("no dataset roots given")
    return out


@dataclass(frozen=True)
class EpisodicDatasetConfig:
    """Configuration for `EpisodicLeRobotDataset`.

    `roots` is one dataset directory or several (multi-task). `dataset_weights`, if
    given, is the per-dataset probability of being drawn for the next episode; it is
    normalized to sum to 1 and must have one entry per root. The default is uniform,
    matching mu-VLA.
    """

    roots: tuple[Path, ...] = field(default=())
    batch_size: int = 8
    action_horizon: int = 5
    seed: int = 42
    rank: int = 0
    world_size: int = 1
    max_episode_steps: int | None = None
    dataset_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", normalize_roots(self.roots))
        if self.dataset_weights is not None:
            weights = tuple(float(w) for w in self.dataset_weights)
            if len(weights) != len(self.roots):
                raise ValueError(
                    f"dataset_weights has {len(weights)} entries but {len(self.roots)} "
                    "dataset roots were given"
                )
            if any(w < 0 for w in weights) or sum(weights) <= 0:
                raise ValueError(f"dataset_weights must be non-negative and not all zero: {weights}")
            object.__setattr__(self, "dataset_weights", weights)


class EpisodicLeRobotDataset(IterableDataset):
    """B time-coherent streams over one or more LeRobot v3 dataset directories."""

    def __init__(self, config: EpisodicDatasetConfig):
        self.config = config
        self.shards = [
            DatasetShard(root, index=i, name=Path(root).name)
            for i, root in enumerate(config.roots)
        ]
        self._validate_shards_are_compatible()

        reference = self.shards[0]
        self.camera_keys = reference.camera_keys
        self.fps = reference.fps
        self.state_dim = reference.state_dim
        self.action_dim = reference.action_dim
        self.total_frames = sum(shard.total_frames for shard in self.shards)

        if config.dataset_weights is None:
            self.weights = np.full(len(self.shards), 1.0 / len(self.shards))
        else:
            raw = np.asarray(config.dataset_weights, dtype=np.float64)
            self.weights = raw / raw.sum()

        logger.info(
            "EpisodicLeRobotDataset: %d dataset(s) [%s], %d episodes, %d frames, "
            "cameras=%s, B=%d, H=%d, weights=%s",
            len(self.shards),
            ", ".join(s.name for s in self.shards),
            self.num_episodes(),
            self.total_frames,
            self.camera_keys,
            config.batch_size,
            config.action_horizon,
            np.round(self.weights, 4).tolist(),
        )

    # --- validation -------------------------------------------------------

    def _validate_shards_are_compatible(self) -> None:
        """Fail early on mixtures whose items cannot be collated into one batch.

        Without this the failure surfaces deep inside the default collate function as
        a stack-shape error that says nothing about which datasets disagree.
        """
        reference = self.shards[0]
        for shard in self.shards[1:]:
            if set(shard.camera_keys) != set(reference.camera_keys):
                raise ValueError(
                    f"camera keys differ: {reference.name} has {sorted(reference.camera_keys)}, "
                    f"{shard.name} has {sorted(shard.camera_keys)}. Multi-task training needs "
                    "the same cameras in every dataset."
                )
            for cam in reference.camera_keys:
                if shard.camera_shapes[cam] != reference.camera_shapes[cam]:
                    raise ValueError(
                        f"camera {cam} has shape {reference.camera_shapes[cam]} in "
                        f"{reference.name} but {shard.camera_shapes[cam]} in {shard.name}; "
                        "frames of different sizes cannot be stacked into one batch."
                    )
            if shard.state_dim != reference.state_dim:
                raise ValueError(
                    f"observation.state dim differs: {reference.name}={reference.state_dim}, "
                    f"{shard.name}={shard.state_dim}"
                )
            if shard.action_dim != reference.action_dim:
                raise ValueError(
                    f"action dim differs: {reference.name}={reference.action_dim}, "
                    f"{shard.name}={shard.action_dim}"
                )
            if shard.fps != reference.fps:
                logger.warning(
                    "fps differs: %s=%d, %s=%d. Mixing control rates changes what one "
                    "memory step means; proceeding anyway.",
                    reference.name,
                    reference.fps,
                    shard.name,
                    shard.fps,
                )

    # --- introspection ----------------------------------------------------

    def num_episodes(self) -> int:
        return sum(shard.num_episodes() for shard in self.shards)

    def episodes_per_dataset(self) -> dict[str, int]:
        return {shard.name: shard.num_episodes() for shard in self.shards}

    @property
    def dataset_names(self) -> list[str]:
        return [shard.name for shard in self.shards]

    # --- single-dataset convenience ---------------------------------------
    #
    # Shortcuts onto the one shard, for single-dataset use (tests, probes). They
    # raise rather than silently answering for shard 0 when several are loaded.

    @property
    def shard(self) -> DatasetShard:
        if len(self.shards) != 1:
            raise AttributeError(
                f"this dataset streams {len(self.shards)} datasets "
                f"({', '.join(self.dataset_names)}); index self.shards explicitly"
            )
        return self.shards[0]

    @property
    def episodes(self):
        return self.shard.episodes

    @property
    def tasks(self) -> list[str]:
        return self.shard.tasks

    @property
    def frame_index(self) -> np.ndarray:
        return self.shard.frame_index

    @property
    def episode_index(self) -> np.ndarray:
        return self.shard.episode_index

    @property
    def video_camera_keys(self) -> list[str]:
        return self.shard.video_camera_keys

    @property
    def image_camera_keys(self) -> list[str]:
        return self.shard.image_camera_keys

    @staticmethod
    def _discover_data_files(root: Path | str) -> list[Path]:
        return DatasetShard._discover_data_files(Path(root))

    def _build_item(self, ep: int, t: int) -> dict:
        return self.shard.build_item(ep, t, self.config.action_horizon)

    # --- streaming --------------------------------------------------------

    def _stream(self, stream_id: int):
        """Infinite generator of consecutive steps, episode after episode.

        The seed depends on the DDP rank and the stream id only, so stream `i` is
        reproducible and independent of every other stream.
        """
        seed = self.config.seed + self.config.rank * 100_003 + stream_id
        rng = np.random.default_rng(seed)
        num_shards = len(self.shards)
        while True:
            shard = self.shards[
                0 if num_shards == 1 else int(rng.choice(num_shards, p=self.weights))
            ]
            ep = int(rng.integers(shard.num_episodes()))
            length = shard.episodes[ep].length
            if self.config.max_episode_steps is not None:
                length = min(length, self.config.max_episode_steps)
            for t in range(length):
                item = shard.build_item(ep, t, self.config.action_horizon)
                if self.config.max_episode_steps is not None and t == length - 1:
                    item["is_last"] = True
                yield item

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError(
                "EpisodicLeRobotDataset must be used with num_workers=0: every worker "
                "would replicate all B streams, so batch[i] would no longer be stream i "
                "and the carried memory state would belong to another trajectory."
            )
        streams = [self._stream(i) for i in range(self.config.batch_size)]
        while True:
            for stream in streams:
                yield next(stream)
