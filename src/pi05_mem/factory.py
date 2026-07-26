"""
factory.py

Build a `PI05MemConfig` / `PI05MemPolicy` / processor triple for a LeRobot v3 dataset
directory, without going through `lerobot-train`'s dataset machinery (which would pull
in torchcodec video decoding we deliberately bypass - see scripts/predecode_videos.py).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PolicyFeature

from .configuration_pi05_mem import PI05MemConfig
from .dataset_stats import load_dataset_stats
from .episodic_dataset import normalize_roots
from .modeling_pi05_mem import PI05MemPolicy
from .processor_pi05_mem import make_pi05_mem_pre_post_processors

logger = logging.getLogger(__name__)

PI05_BASE_REPO = "lerobot/pi05_base"


def features_from_dataset(root: Path | str) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    """Derive policy input/output features from `meta/info.json`."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())

    input_features: dict[str, PolicyFeature] = {}
    for key, feature in info["features"].items():
        # "video" is MIKASA-Robo (mp4 + a pre-decoded .npy cache), "image" is LIBERO
        # (PNG bytes inline in the data parquet). Both declare HWC shapes.
        if feature["dtype"] in ("video", "image"):
            h, w, c = feature["shape"]
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
        elif key == "observation.state":
            input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=tuple(feature["shape"]))

    action_shape = tuple(info["features"]["action"]["shape"])
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=action_shape)}
    return input_features, output_features


def features_from_datasets(
    roots: Path | str | Sequence[Path | str],
) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    """Features of a multi-task mixture: identical in every dataset, or it is an error.

    One policy has one input/output signature, so a mixture whose datasets disagree on
    camera set, image size, state dim or action dim cannot be trained as one model.
    `EpisodicLeRobotDataset` checks the same thing on the data side; this check runs
    earlier, before any weights are built.
    """
    paths = normalize_roots(roots)
    reference = features_from_dataset(paths[0])
    for path in paths[1:]:
        other = features_from_dataset(path)
        if other != reference:
            mismatched = {
                key
                for features_a, features_b in zip(reference, other, strict=True)
                for key in set(features_a) | set(features_b)
                if features_a.get(key) != features_b.get(key)
            }
            raise ValueError(
                f"{paths[0].name} and {path.name} declare different policy features "
                f"for {sorted(mismatched)}; they cannot be trained as one multi-task model"
            )
    return reference


def make_config(
    root: Path | str | Sequence[Path | str],
    *,
    chunk_size: int = 5,
    n_action_steps: int = 1,
    device: str = "cuda",
    dtype: str = "bfloat16",
    **memory_kwargs: Any,
) -> PI05MemConfig:
    """Create a PI05MemConfig wired to one dataset directory or a multi-task mixture.

    `memory_kwargs` are the mu-VLA-named memory flags: use_memory, num_mem_tokens,
    memory_update, ema_alpha, tbptt_length, attention_mask_mode, memory_write_scale,
    memory_log_freq, memory_expensive_log_freq.
    """
    input_features, output_features = features_from_datasets(root)
    config = PI05MemConfig(
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        device=device,
        dtype=dtype,
        **memory_kwargs,
    )
    config.input_features = input_features
    config.output_features = output_features
    # normalization_mapping is left at the PI05 default (VISUAL: IDENTITY, STATE and
    # ACTION: QUANTILES). Pixels are normalized by pi0.5 itself inside
    # _preprocess_images, and the discrete state token assumes quantile-normalized
    # state in [-1, 1], which is what the pretrained checkpoint was trained with.
    config.validate_features()
    return config


def make_policy(
    config: PI05MemConfig,
    *,
    pretrained: str | Path | None = PI05_BASE_REPO,
) -> PI05MemPolicy:
    """Instantiate the policy, optionally initialising from a pi0.5 checkpoint.

    `strict=False`: the pretrained pi0.5 checkpoint has no `memory_module`, which is
    trained from scratch.
    """
    if pretrained is None:
        # `from_pretrained` moves the policy itself; the scratch path has to do it here.
        return PI05MemPolicy(config).to(config.device)

    policy = PI05MemPolicy.from_pretrained(pretrained, config=config, strict=False)
    return policy


def make_processors(
    config: PI05MemConfig,
    root: Path | str | Sequence[Path | str],
    tokenizer_name: str | Path | None = None,
):
    """Pre/post processors. For several roots the statistics are the pooled ones."""
    stats = load_dataset_stats(normalize_roots(root))
    return make_pi05_mem_pre_post_processors(config, dataset_stats=stats, tokenizer_name=tokenizer_name)


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
