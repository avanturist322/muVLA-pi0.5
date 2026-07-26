# Derived from HuggingFace LeRobot (https://github.com/huggingface/lerobot),
# licensed under the Apache License, Version 2.0.
#
#     Copyright 2024-2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications for pi05-mem: Copyright (c) 2026 Egor Cherepanov, MIT-licensed
# (see LICENSE), applied on top of the Apache-2.0 original.

"""
configuration_pi05_mem.py

Config for pi05-mem = LeRobot's PI05 + the mu-VLA recurrent memory mechanism.

Flag names and defaults intentionally mirror mu-VLA's `FinetuneConfig`
(`vla-scripts/finetune.py`) so that a run configured for mu-VLA maps onto
pi05-mem without translation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

logger = logging.getLogger(__name__)

MEMORY_UPDATE_RULES = ("tbptt", "ema")
ATTENTION_MASK_MODES = ("custom", "full")


@PreTrainedConfig.register_subclass("pi05_mem")
@dataclass
class PI05MemConfig(PI05Config):
    """PI05 with mu-VLA-style recurrent memory tokens.

    With `use_memory=False` this is behaviourally identical to plain `PI05Config`:
    no MEM tokens are inserted and no memory module is built.
    """

    # --- pi05-mem recurrent memory (names mirror mu-VLA) ---
    use_memory: bool = False
    """If True, enables recurrent memory tokens injected into the VLM prefix."""

    num_mem_tokens: int = 4
    """Number of MEM tokens injected into the prefix. mu-VLA experiments used 64."""

    memory_update: str = "tbptt"
    """Cross-step memory update rule.

    "tbptt": keep the graph within a `tbptt_length` window, detach at the boundary.
    "ema":   no cross-step gradient flow; M_in[t+1] = alpha*M_out[t] + (1-alpha)*M_in[t].
    """

    ema_alpha: float = 0.1
    """Mixing coefficient for memory_update="ema". alpha=1 recovers TBPTT-length-1;
    alpha=0 freezes memory. Ignored when memory_update="tbptt"."""

    tbptt_length: int = 1
    """TBPTT truncation length K: gradients flow through this many env steps.
    Ignored when memory_update="ema"."""

    attention_mask_mode: str = "custom"
    """Attention mask used when memory is active.

    "custom" - MEM tokens live in the prefix block, so (like every other prefix
        token) they cannot attend to the action suffix. This is the pi0.5-native
        block structure and the analogue of mu-VLA's context/action split mask.
        It is also the only mode compatible with prefix KV-caching at inference.
    "full"   - ablation: MEM rows are additionally allowed to attend to the action
        suffix. Breaks the prefix/suffix insulation, so inference must recompute
        the joint forward at every denoising step (no prefix cache). Provided for
        parity with mu-VLA's `attention_mask_mode` ablation; not recommended,
        because under flow matching the suffix carries pure noise at t=1.

    Saved to memory_meta.json so evaluation auto-detects the right mode.
    """

    memory_write_scale: float = 5.8
    """Scalar applied to the MEM hidden-state readout before it becomes the next
    step's MEM input embedding.

    mu-VLA needed no such factor: in Prismatic/Llama, input embeddings and final
    hidden states share a scale (both ~0.02). PaliGemma does not - lerobot's
    `pi_gemma.py` drops the Gemma sqrt(width) input multiplier, so measured on
    `lerobot/pi05_base`:

        embed_image std           4.16
        embed_language_tokens std 9.71
        prefix output std         1.19   -> ratio 5.82

    Default 5.8 puts the MEM readout in the same range as the other prefix input
    embeddings. Set to 1.0 for verbatim mu-VLA feedback; re-measure for another
    backbone with `scripts/probe_embed_scale.py`.
    """

    memory_init_std: float = 4.0
    """Std of `MemoryModule.initial_memory`. mu-VLA used 0.02 to match Llama's input
    embeddings; 4.0 matches PaliGemma's image-embedding scale. Note RMSNorm makes the
    attention path scale-invariant per token, so this mainly affects the residual
    stream contribution of the MEM positions."""

    receding_horizon: bool | None = None
    """Whether inference queries the model at *every* environment step (mu-VLA's
    `receding_horizon`). `None` means auto: enabled whenever `use_memory=True`.

    This matters because memory is recurrent. Training rolls the state forward one
    env step per forward pass, so evaluation must do the same. LeRobot's
    `select_action` only calls `predict_action_chunk` when its action queue drains,
    i.e. once every `n_action_steps` steps -- with `n_action_steps=8` the memory would
    advance 8x slower than in training and immediately go out of distribution.
    Receding horizon therefore requires `n_action_steps == 1`: predict a chunk, execute
    only its first action, re-query (and re-update memory) next step.

    Saved to memory_meta.json so evaluation auto-detects the right mode."""

    memory_log_freq: int = 0
    """Lightweight memory diagnostics (norms, grad norms) every N gradient steps.
    0 = disabled. Only active when use_memory=True."""

    memory_expensive_log_freq: int = 0
    """Expensive memory diagnostics (per-token drift, attention mass) every N
    gradient steps. 0 = disabled."""

    def __post_init__(self):
        super().__post_init__()

        if self.memory_update not in MEMORY_UPDATE_RULES:
            raise ValueError(
                f"Invalid memory_update: {self.memory_update!r} (must be one of {MEMORY_UPDATE_RULES})"
            )
        if self.attention_mask_mode not in ATTENTION_MASK_MODES:
            raise ValueError(
                f"Invalid attention_mask_mode: {self.attention_mask_mode!r} "
                f"(must be one of {ATTENTION_MASK_MODES})"
            )
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in [0, 1], got {self.ema_alpha}")
        if self.tbptt_length < 1:
            raise ValueError(f"tbptt_length must be >= 1, got {self.tbptt_length}")
        if self.use_memory and self.num_mem_tokens < 1:
            raise ValueError(f"use_memory=True requires num_mem_tokens >= 1, got {self.num_mem_tokens}")

        if self.use_memory and self.receding_horizon is False:
            logger.warning(
                "use_memory=True with receding_horizon=False: memory will advance once per "
                "action chunk instead of once per env step, which does not match training. "
                "This is an ablation, not a normal setting."
            )
        elif self.effective_receding_horizon and self.n_action_steps != 1:
            logger.warning(
                "receding_horizon is active (use_memory=True) but n_action_steps=%d. "
                "Rollouts must use n_action_steps=1; %s.select_action will refuse otherwise.",
                self.n_action_steps,
                type(self).__name__,
            )

    @property
    def effective_tbptt_length(self) -> int:
        """Number of env steps per backward pass. EMA never builds a cross-step graph."""
        if not self.use_memory:
            return 1
        return 1 if self.memory_update == "ema" else self.tbptt_length

    @property
    def effective_receding_horizon(self) -> bool:
        """Resolve `receding_horizon=None` the way mu-VLA does: on iff memory is on."""
        if self.receding_horizon is None:
            return bool(self.use_memory)
        return bool(self.receding_horizon)
