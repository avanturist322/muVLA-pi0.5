"""
frame_sources.py

Where `EpisodicLeRobotDataset` gets pixels from.

LeRobot v3 stores camera frames one of two ways, and the two datasets we train on use
one each:

* `dtype="image"` - PNG bytes embedded in the data parquet as
  `struct<bytes: binary, path: string>`. This is `lerobot/libero_spatial_image`.
  Decoded on demand; a 256x256 PNG decodes in well under a millisecond, and keeping
  the compressed bytes costs ~10x less RAM than keeping decoded frames.
* `dtype="video"` - frames live in mp4 files. torchcodec is unusable in this container
  (see scripts/predecode_videos.py), so the frames are decoded once into
  `cache/<camera>.npy` and memory-mapped. This is the MIKASA-Robo export.

Both expose the same `frame(row) -> HWC uint8` interface so the dataset does not care.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class FrameSource(Protocol):
    """Random access to one camera's frames by flat row index."""

    def frame(self, row: int) -> np.ndarray:
        """Return frame `row` as an HWC uint8 array."""

    def __len__(self) -> int: ...


class NpyCacheFrameSource:
    """Frames from a pre-decoded `.npy` cache, memory-mapped.

    Used for `dtype="video"` cameras. Row indices are positions in the flat frame
    order of the dataset, which is how `scripts/predecode_videos.py` writes the cache.
    """

    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"missing frame cache {path}; run scripts/predecode_videos.py"
            )
        self.path = path
        self.array = np.load(path, mmap_mode="r")

    def frame(self, row: int) -> np.ndarray:
        return np.asarray(self.array[row])

    def __len__(self) -> int:
        return len(self.array)


class ParquetImageFrameSource:
    """Frames from PNG/JPEG bytes embedded in the data parquet.

    Used for `dtype="image"` cameras. `blobs[row]` is the encoded image for row `row`.
    """

    def __init__(self, blobs: list[bytes], *, expected_shape: tuple[int, int, int] | None = None):
        self._blobs = blobs
        self._expected_shape = expected_shape

    def frame(self, row: int) -> np.ndarray:
        from PIL import Image

        with Image.open(io.BytesIO(self._blobs[row])) as img:
            array = np.asarray(img.convert("RGB"))
        if self._expected_shape is not None and array.shape != self._expected_shape:
            raise ValueError(
                f"decoded frame has shape {array.shape}, expected {self._expected_shape}"
            )
        return array

    def __len__(self) -> int:
        return len(self._blobs)


def extract_image_blobs(column) -> list[bytes]:
    """Pull the raw encoded bytes out of a LeRobot image column.

    The column is `struct<bytes: binary, path: string>`; some exports store a plain
    binary column instead, so accept both.
    """
    values = column.to_pylist()
    if not values:
        return []
    if isinstance(values[0], dict):
        return [v["bytes"] for v in values]
    if isinstance(values[0], (bytes, bytearray)):
        return [bytes(v) for v in values]
    raise TypeError(f"unsupported image column element type: {type(values[0])}")
