"""
memory_meta.py

Persist / auto-detect the memory configuration of a checkpoint.

Port of mu-VLA's `memory_meta.json` + `detect_memory_config()`
(`experiments/robot/openvla_utils.py:519`): evaluation code should not have to be
told whether a checkpoint uses memory - it reads it off the checkpoint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from .configuration_pi05_mem import PI05MemConfig

logger = logging.getLogger(__name__)

MEMORY_META_FILENAME = "memory_meta.json"
MEMORY_MODULE_FILENAME = "memory_module.pt"

_META_FIELDS = (
    "use_memory",
    "num_mem_tokens",
    "memory_update",
    "ema_alpha",
    "tbptt_length",
    "attention_mask_mode",
    "memory_write_scale",
    "memory_init_std",
    "receding_horizon",
)


def memory_meta_from_config(config: PI05MemConfig) -> dict[str, Any]:
    return {field: getattr(config, field) for field in _META_FIELDS}


def save_memory_meta(directory: str | Path, config: PI05MemConfig) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MEMORY_META_FILENAME
    path.write_text(json.dumps(memory_meta_from_config(config), indent=2))
    return path


def detect_memory_config(directory: str | Path) -> dict[str, Any]:
    """Resolve a checkpoint's memory settings.

    Order, mirroring mu-VLA:
      1. `memory_meta.json` if present;
      2. otherwise peek at the saved memory module and infer `num_mem_tokens`;
      3. otherwise assume memory is disabled.
    """
    directory = Path(directory)

    meta_path = directory / MEMORY_META_FILENAME
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            logger.info("detected memory config from %s: %s", meta_path, meta)
            return meta
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read %s (%s); falling back to state-dict probe", meta_path, exc)

    module_path = directory / MEMORY_MODULE_FILENAME
    if module_path.exists():
        try:
            state = torch.load(module_path, map_location="cpu", weights_only=True)
            initial = state.get("initial_memory")
            if initial is not None:
                meta = {"use_memory": True, "num_mem_tokens": int(initial.shape[0])}
                logger.info("inferred memory config from %s: %s", module_path, meta)
                return meta
        except (OSError, RuntimeError) as exc:
            logger.warning("could not probe %s (%s)", module_path, exc)

    return {"use_memory": False}


def save_memory_module(directory: str | Path, model, config: PI05MemConfig) -> None:
    """Save the memory module alongside `memory_meta.json`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_memory_meta(directory, config)
    if getattr(model, "memory_module", None) is not None:
        torch.save(model.memory_module.state_dict(), directory / MEMORY_MODULE_FILENAME)
