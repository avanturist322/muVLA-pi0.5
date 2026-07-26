"""
dataset_stats.py

Normalization statistics for one or several LeRobot v3 dataset roots.

pi0.5 normalizes `observation.state` and `action` with QUANTILES
(`2*(x - q01)/(q99 - q01) - 1`), and the normalized state is then discretized into
256 bins and written into the *text prompt*. Wrong quantiles therefore do not merely
rescale a tensor, they change the prompt the model reads - so for multi-task training
the statistics have to describe the actual mixture, not one of its parts.

Single root: `meta/stats.json` is used verbatim, exactly as before multi-task support
existed, so single-dataset runs are bit-identical to earlier ones.

Several roots: `observation.state` and `action` statistics are recomputed *exactly*
from the raw parquet columns of all roots pooled together (mean/std/min/max and the
q01/q10/q50/q90/q99 quantiles). Per-dataset quantiles cannot be merged analytically,
which is why the raw columns are re-read. The result is cached next to the datasets,
keyed by the sorted dataset names, so the pass happens once per mixture.

Everything else (camera statistics and bookkeeping columns) is copied from the first
root: cameras are normalized with IDENTITY in `PI05MemConfig.normalization_mapping`,
so pi0.5 never consumes those entries, and the bookkeeping columns are not policy
features at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .shard import DatasetShard

logger = logging.getLogger(__name__)

# The features pi0.5 actually normalizes with QUANTILES; the ones worth pooling exactly.
POOLED_KEYS = ("observation.state", "action")

# Quantile levels LeRobot writes into meta/stats.json.
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def _to_tensors(raw: dict) -> dict[str, dict[str, torch.Tensor]]:
    """JSON stats -> the {feature: {stat: tensor}} mapping the processors expect.

    `count` is dropped (it is a sample count, not a statistic over the feature) and
    keys starting with `_` are metadata we add ourselves, not features.
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    for feature, entry in raw.items():
        if feature.startswith("_"):
            continue
        out[feature] = {
            stat: torch.tensor(value, dtype=torch.float32)
            for stat, value in entry.items()
            if stat != "count"
        }
    return out


def load_dataset_stats(
    roots: Path | str | Sequence[Path | str],
) -> dict[str, dict[str, torch.Tensor]]:
    """Statistics for one dataset root, or the pooled statistics for several."""
    if isinstance(roots, (str, Path)):
        paths = [Path(roots)]
    else:
        paths = [Path(r) for r in roots]
    if not paths:
        raise ValueError("no dataset roots given")

    if len(paths) == 1:
        return _to_tensors(json.loads((paths[0] / "meta" / "stats.json").read_text()))

    return _to_tensors(combined_stats(paths))


# --- combined statistics ------------------------------------------------


def _cache_path(paths: Sequence[Path]) -> Path:
    names = sorted(p.name for p in paths)
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
    resolved = [p.resolve() for p in paths]
    try:
        parent = Path(os.path.commonpath([str(p) for p in resolved]))
        if parent in resolved:  # one root is a parent of the others
            parent = parent.parent
    except ValueError:  # different drives / no common path
        parent = resolved[0].parent
    return parent / f"_combined_stats_{digest}.json"


def combined_stats(paths: Sequence[Path], *, use_cache: bool = True) -> dict:
    """Pooled statistics over several dataset roots, as a JSON-shaped dict."""
    names = sorted(p.name for p in paths)
    cache = _cache_path(paths)

    if use_cache and cache.exists():
        try:
            cached = json.loads(cache.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning("combined stats: %s is unreadable (%s), recomputing", cache, exc)
            cached = {}
        if cached.get("_provenance", {}).get("datasets") == names:
            logger.info("combined stats: reusing cache %s", cache)
            return cached
        logger.warning("combined stats: %s does not match %s, recomputing", cache, names)

    logger.info("combined stats: pooling %s from raw parquet columns", names)
    columns = {key: [] for key in POOLED_KEYS}
    counts: dict[str, int] = {}
    for path in paths:
        rows = _read_columns(path, POOLED_KEYS)
        counts[path.name] = len(rows[POOLED_KEYS[0]])
        for key in POOLED_KEYS:
            columns[key].append(rows[key])

    stats = json.loads((Path(paths[0]) / "meta" / "stats.json").read_text())
    for key in POOLED_KEYS:
        stacked = np.concatenate(columns[key], axis=0)
        stats[key] = _describe(stacked)

    stats["_provenance"] = {
        "datasets": names,
        "rows_per_dataset": counts,
        "pooled_keys": list(POOLED_KEYS),
        "copied_from": Path(paths[0]).name,
        "note": (
            "observation.state and action recomputed from raw rows present on disk; "
            "all other entries copied from copied_from (unused: cameras are IDENTITY-normalized)"
        ),
    }

    if use_cache:
        try:
            # Temp file + rename, and the temp name carries the pid: the eval suite
            # starts one process per GPU at the same second, and they all compute this
            # cache. A plain write_text lets a reader see a half-written file, which
            # would silently normalize the state with garbage quantiles - and the
            # normalized state goes into the prompt as 256 discrete bins.
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(stats, indent=2))
            os.replace(tmp, cache)
            logger.info("combined stats: cached to %s", cache)
        except OSError as exc:
            logger.warning("combined stats: could not write cache %s: %s", cache, exc)
    return stats


def _read_columns(root: Path, keys: Sequence[str]) -> dict[str, np.ndarray]:
    """Read `keys` from every data parquet present under `root`, concatenated."""
    import pyarrow.parquet as pq

    chunks: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for path in DatasetShard._discover_data_files(root):
        table = pq.read_table(path, columns=list(keys))
        for key in keys:
            chunks[key].append(np.asarray(table.column(key).to_pylist(), dtype=np.float64))
    return {key: np.concatenate(parts, axis=0) for key, parts in chunks.items()}


def _describe(values: np.ndarray) -> dict:
    """Per-dimension statistics in the shape LeRobot's stats.json uses."""
    stats = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }
    for name, q in QUANTILES.items():
        stats[name] = np.quantile(values, q, axis=0).tolist()
    return stats
