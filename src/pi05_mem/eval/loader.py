"""
loader.py

Rebuild a trained pi05-mem policy for rollout, the way mu-VLA's eval scripts do:
the *checkpoint* says whether memory is on and how it is configured, not the CLI.

`memory_meta.json` (written next to every checkpoint by `train._save`) is the source
of truth; `config.json` supplies chunk_size; the dataset directory supplies the
normalization statistics and the feature layout, because those describe the data the
policy was trained on and there is nowhere else to get them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ..configuration_pi05_mem import PI05MemConfig
from ..episodic_dataset import normalize_roots
from ..factory import make_config, make_policy, make_processors
from ..memory_meta import MEMORY_MODULE_FILENAME, detect_memory_config
from ..modeling_pi05_mem import PI05MemPolicy
from .integrity import verify_checkpoint_loaded, verify_eval_datasets

logger = logging.getLogger(__name__)

MEMORY_FLAGS = (
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


@dataclass
class EvalBundle:
    """Everything a rollout needs, kept together so the runners stay short."""

    policy: PI05MemPolicy
    config: PI05MemConfig
    preprocessor: Any
    postprocessor: Any
    memory_meta: dict[str, Any]

    @property
    def uses_memory(self) -> bool:
        return bool(self.config.use_memory)

    @property
    def camera_keys(self) -> list[str]:
        from lerobot.configs.types import FeatureType

        return [k for k, f in self.config.input_features.items() if f.type == FeatureType.VISUAL]


def _chunk_size_from_checkpoint(checkpoint: Path, fallback: int) -> int:
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        logger.warning("no config.json in %s; assuming chunk_size=%d", checkpoint, fallback)
        return fallback
    saved = json.loads(config_path.read_text())
    return int(saved.get("chunk_size", fallback))


def load_memory_module(checkpoint: Path, policy: PI05MemPolicy) -> None:
    """Load `memory_module.pt` explicitly.

    `save_pretrained` already puts the memory parameters into model.safetensors, but
    that path goes through PI05's key remapper with `strict=False`, so a rename would
    silently leave `initial_memory` at its random init - and an eval that starts from
    random memory looks like "memory does not help" rather than like a bug. Loading
    the standalone file with `strict=True` makes that failure loud.
    """
    module = getattr(policy.model, "memory_module", None)
    if module is None:
        return
    path = Path(checkpoint) / MEMORY_MODULE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{checkpoint} enables memory but has no {MEMORY_MODULE_FILENAME}; "
            "the checkpoint was not written by pi05_mem.train"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    module.load_state_dict(state, strict=True)
    module.to(policy.config.device)
    logger.info(
        "loaded %s: initial_memory norm=%.4f",
        path.name,
        module.initial_memory.detach().float().norm().item(),
    )


def resolve_inference_regime(
    explicit: bool | None, memory_kwargs: dict, chunk_size: int
) -> tuple[bool, int]:
    """Which inference regime this eval runs, and the `n_action_steps` that implements it.

    The two regimes differ in exactly this number, so it is derived from the flag rather
    than pinned. Hardcoding 1 made `--receding-horizon false` a no-op and turned the
    standard-inference run into a duplicate of the receding-horizon one - two identical
    columns in the results table, reported as an ablation.

      receding horizon   n_action_steps=1           requery every env step, execute
                                                    chunk[0]. Memory advances once per
                                                    env step, exactly as in training.
      standard (chunked) n_action_steps=chunk_size  requery when the queue drains,
                                                    execute the chunk open loop. This is
                                                    mu-VLA's baseline regime
                                                    (run_mikasa_robo_eval.py:551,
                                                    deque(maxlen=num_open_loop_steps)
                                                    with num_open_loop_steps ==
                                                    NUM_ACTIONS_CHUNK).

    `explicit=None` is the CLI's `auto`: take the regime the checkpoint recorded, and
    fall back to "on iff this checkpoint uses memory" (mu-VLA's rule) for checkpoints
    that recorded nothing. `select_action` refuses n_action_steps != 1 while receding
    horizon is active, so the two cannot be silently mixed.
    """
    receding = explicit
    if receding is None:
        saved = memory_kwargs.get("receding_horizon")
        receding = (
            bool(saved) if saved is not None
            else bool(memory_kwargs.get("use_memory", False))
        )
    return receding, (1 if receding else chunk_size)


def load_eval_policy(
    checkpoint: str | Path,
    data: str | Path | Sequence[str | Path],
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    chunk_size: int | None = None,
    receding_horizon: bool | None = None,
    tokenizer_name: str | Path | None = None,
) -> EvalBundle:
    """Rebuild policy + processors from a checkpoint directory.

    `data` may be several dataset directories. For a multi-task checkpoint it must be
    the *same* set the model was trained on: pi0.5 discretizes the normalized state
    into 256 bins inside the text prompt, so evaluating with one task's quantiles when
    the model was trained on the pooled ones feeds it a corrupted prompt.
    """
    checkpoint = Path(checkpoint)

    memory_meta = detect_memory_config(checkpoint)
    memory_kwargs = {k: v for k, v in memory_meta.items() if k in MEMORY_FLAGS}
    if receding_horizon is not None:
        memory_kwargs["receding_horizon"] = receding_horizon

    resolved_chunk = chunk_size or _chunk_size_from_checkpoint(checkpoint, fallback=5)
    receding, n_action_steps = resolve_inference_regime(
        receding_horizon, memory_kwargs, resolved_chunk
    )

    config = make_config(
        data,
        chunk_size=resolved_chunk,
        n_action_steps=n_action_steps,
        device=device,
        dtype=dtype,
        **memory_kwargs,
    )

    dataset_check = verify_eval_datasets(checkpoint, normalize_roots(data))
    policy = make_policy(config, pretrained=checkpoint)
    verify_checkpoint_loaded(checkpoint, policy)
    if config.use_memory:
        load_memory_module(checkpoint, policy)
    policy.eval()

    preprocessor, postprocessor = make_processors(config, data, tokenizer_name=tokenizer_name)

    logger.info(
        "eval policy: memory=%s update=%s mem_tokens=%s mask=%s chunk=%d receding=%s "
        "n_action_steps=%d (%s) datasets_verified=%s",
        config.use_memory,
        config.memory_update,
        config.num_mem_tokens,
        config.attention_mask_mode,
        config.chunk_size,
        config.effective_receding_horizon,
        config.n_action_steps,
        "requery every env step" if n_action_steps == 1 else f"open loop, {n_action_steps} actions",
        dataset_check["verified"],
    )
    return EvalBundle(
        policy=policy,
        config=config,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        memory_meta=memory_meta,
    )
