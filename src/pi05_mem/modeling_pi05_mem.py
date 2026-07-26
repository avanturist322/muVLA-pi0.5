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
modeling_pi05_mem.py

PI05 + mu-VLA recurrent memory tokens.

Where the MEM tokens live
-------------------------
pi0.5 splits its context into a *prefix* (images + language, run through the
PaliGemma 2B backbone) and a *suffix* (noisy action chunk, run through the
300M action expert). The two blocks are joined by a block attention mask in which
the prefix is fully bidirectional and cannot see the suffix, while the suffix sees
everything.

mu-VLA injects MEM tokens right after the vision/proprio embeddings and before the
language tokens, and reads the next memory state off the corresponding hidden
states. We do exactly the same, in the prefix:

    [ img_0 ... img_N | MEM_0 ... MEM_{m-1} | lang_0 ... lang_L ] [ action_0 ... ]
                        ^^^^^^^^^^^^^^^^^^^
                        injected here; read back from the prefix output

Because the MEM tokens sit inside the prefix block, the stock pi0.5 mask already
implements mu-VLA's `attention_mask_mode="custom"` invariant: context tokens do not
attend to action tokens, action tokens attend to everything. `"full"` is the
ablation that additionally lets the MEM rows attend to the action suffix.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import require_package

from .configuration_pi05_mem import PI05MemConfig
from .memory import MemoryModule

logger = logging.getLogger(__name__)


class PI05MemPytorch(PI05Pytorch):
    """PI05Pytorch with recurrent MEM tokens in the VLM prefix.

    Note the changed signatures relative to the base class: `forward` returns
    `(losses, new_mem_state)` instead of `losses`. `PI05MemPolicy` is aware of this;
    do not mix this module with the stock `PI05Policy`.
    """

    def __init__(self, config: PI05MemConfig, rtc_processor: Any | None = None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.config: PI05MemConfig = config

        self.memory_module: MemoryModule | None = None
        if config.use_memory:
            width = self.paligemma_with_expert.paligemma.model.language_model.get_input_embeddings().weight.shape[1]
            self.memory_module = MemoryModule(
                config.num_mem_tokens, width, init_std=config.memory_init_std
            )
            logger.info(
                "pi05-mem: %d MEM tokens of width %d, update=%s, mask=%s",
                config.num_mem_tokens,
                width,
                config.memory_update,
                config.attention_mask_mode,
            )

    # --- memory helpers -------------------------------------------------

    @property
    def uses_memory(self) -> bool:
        return self.memory_module is not None

    def get_initial_memory(self, batch_size: int) -> Tensor | None:
        if not self.uses_memory:
            return None
        return self.memory_module.get_initial_state(batch_size)

    def _read_memory(self, prefix_out: Tensor, mem_slice: tuple[int, int]) -> Tensor:
        start, end = mem_slice
        new_mem = prefix_out[:, start:end, :]
        if self.config.memory_write_scale != 1.0:
            new_mem = new_mem * self.config.memory_write_scale
        return new_mem.to(dtype=torch.float32)

    # --- prefix embedding with MEM injection ----------------------------

    def embed_prefix_mem(
        self, images, img_masks, tokens, masks, mem_state: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, tuple[int, int] | None]:
        """Same as `PI05Pytorch.embed_prefix`, with MEM tokens between images and language.

        Returns `(embs, pad_masks, att_masks, mem_slice)` where `mem_slice` indexes the
        MEM positions inside the prefix (or None when memory is disabled).
        """
        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        att_masks: list[int] = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        mem_slice: tuple[int, int] | None = None
        if self.uses_memory:
            if mem_state is None:
                raise ValueError("use_memory=True but mem_state is None; call get_initial_memory() first")
            num_mem = mem_state.shape[1]
            if num_mem != self.config.num_mem_tokens:
                raise ValueError(
                    f"mem_state has {num_mem} tokens, config.num_mem_tokens={self.config.num_mem_tokens}"
                )
            mem_start = sum(e.shape[1] for e in embs)
            mem_slice = (mem_start, mem_start + num_mem)

            mem_emb = mem_state.to(dtype=embs[0].dtype, device=embs[0].device)
            bsize = mem_emb.shape[0]
            embs.append(mem_emb)
            # MEM tokens are always present -> never padded, and stay inside the
            # bidirectional prefix block (att_masks entry 0 = "same block as previous").
            pad_masks.append(torch.ones(bsize, num_mem, dtype=torch.bool, device=mem_emb.device))
            att_masks += [0] * num_mem

        def lang_embed_func(tokens):
            return self.paligemma_with_expert.embed_language_tokens(tokens)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks_t = att_masks_t[None, :].expand(pad_masks.shape[0], len(att_masks))

        return embs, pad_masks, att_masks_t, mem_slice

    def embed_prefix(self, images, img_masks, tokens, masks):
        """Base-class-compatible entry point; only valid when memory is disabled."""
        if self.uses_memory:
            raise RuntimeError("use embed_prefix_mem() when use_memory=True")
        embs, pad_masks, att_masks, _ = self.embed_prefix_mem(images, img_masks, tokens, masks)
        return embs, pad_masks, att_masks

    def _open_mem_rows_to_suffix(
        self,
        att_2d_masks: Tensor,
        mem_slice: tuple[int, int],
        prefix_len: int,
        suffix_pad_masks: Tensor,
    ) -> Tensor:
        """attention_mask_mode="full": let MEM rows attend to the action suffix."""
        start, end = mem_slice
        att_2d_masks = att_2d_masks.clone()
        att_2d_masks[:, start:end, prefix_len:] = suffix_pad_masks[:, None, :]
        return att_2d_masks

    # --- training -------------------------------------------------------

    def forward(  # type: ignore[override]
        self, images, img_masks, tokens, masks, actions, noise, time, mem_state: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Training forward. Returns `(per-element flow-matching losses, new_mem_state)`."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, mem_slice = self.embed_prefix_mem(
            images, img_masks, tokens, masks, mem_state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        if mem_slice is not None and self.config.attention_mask_mode == "full":
            att_2d_masks = self._open_mem_rows_to_suffix(
                att_2d_masks, mem_slice, prefix_pad_masks.shape[1], suffix_pad_masks
            )

        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        new_mem_state = None if mem_slice is None else self._read_memory(prefix_out, mem_slice)

        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none"), new_mem_state

    # --- inference ------------------------------------------------------

    @torch.no_grad()
    def sample_actions_mem(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        mem_state: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        """Inference forward. Returns `(actions, new_mem_state)`."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks, mem_slice = self.embed_prefix_mem(
            images, img_masks, tokens, masks, mem_state
        )

        if mem_slice is not None and self.config.attention_mask_mode == "full":
            return self._sample_actions_joint(
                prefix_embs, prefix_pad_masks, prefix_att_masks, mem_slice, noise, num_steps, **kwargs
            )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        new_mem_state = None if mem_slice is None else self._read_memory(prefix_out, mem_slice)

        actions = euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=current_timestep,
            ),
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )
        return actions, new_mem_state

    def _sample_actions_joint(
        self, prefix_embs, prefix_pad_masks, prefix_att_masks, mem_slice, noise, num_steps, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        """Non-cached denoising for attention_mask_mode="full".

        With the MEM rows attending to the suffix, the prefix keys/values change at
        every denoising step, so the prefix KV cache is invalid and the joint forward
        must be recomputed `num_steps` times. This path exists for ablation parity with
        mu-VLA and is ~num_steps times more expensive than the cached one.
        """
        last_prefix_out: dict[str, Tensor] = {}
        prefix_len = prefix_pad_masks.shape[1]

        def joint_denoise(x_t, timestep):
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
            embs_p = prefix_embs
            if (
                self.paligemma_with_expert.paligemma.model.language_model.layers[
                    0
                ].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                embs_p = embs_p.to(dtype=torch.bfloat16)
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            att_2d_masks = self._open_mem_rows_to_suffix(
                att_2d_masks, mem_slice, prefix_len, suffix_pad_masks
            )
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=prepare_attention_masks_4d(att_2d_masks),
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[embs_p, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            last_prefix_out["value"] = prefix_out
            return self.action_out_proj(suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32))

        actions = euler_integrate(
            joint_denoise,
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )
        new_mem_state = self._read_memory(last_prefix_out["value"], mem_slice)
        return actions, new_mem_state


class PI05MemPolicy(PI05Policy):
    """PI05Policy that carries a recurrent memory state across steps."""

    config_class = PI05MemConfig
    name = "pi05_mem"

    def __init__(self, config: PI05MemConfig, **kwargs):
        # Mirrors PI05Policy.__init__ with PI05Pytorch swapped for PI05MemPytorch;
        # calling super() would build the 3B backbone twice.
        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = PI05MemPytorch(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    # --- memory state carried across env steps at inference --------------

    def reset(self):
        super().reset()
        self._mem_state: Tensor | None = None

    @property
    def uses_memory(self) -> bool:
        return self.model.uses_memory

    def get_initial_memory(self, batch_size: int) -> Tensor | None:
        return self.model.get_initial_memory(batch_size)

    # --- training --------------------------------------------------------

    def forward_with_memory(
        self, batch: dict[str, Tensor], mem_state: Tensor | None = None, reduction: str = "mean"
    ) -> tuple[Tensor, dict, Tensor | None]:
        """Like `PI05Policy.forward`, but threads the recurrent memory state through."""
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.prepare_action(batch)
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        losses, new_mem_state = self.model.forward(
            images, img_masks, tokens, masks, actions, noise, time, mem_state=mem_state
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {"loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist()}

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict, new_mem_state

        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict, new_mem_state

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Stateless entry point (memory starts from `initial_memory` every call).

        Sequential training must use `forward_with_memory`; this exists so that the
        policy stays a drop-in `PreTrainedPolicy` for non-recurrent tooling.
        """
        mem_state = self.get_initial_memory(batch[ACTION].shape[0]) if self.uses_memory else None
        if mem_state is not None:
            mem_state = mem_state.to(self.config.device)
        loss, loss_dict, _ = self.forward_with_memory(batch, mem_state=mem_state, reduction=reduction)
        return loss, loss_dict

    # --- inference -------------------------------------------------------

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Same as the base implementation, but refuses chunked execution with memory.

        `PI05Policy.select_action` keeps a `deque(maxlen=n_action_steps)` and only calls
        `predict_action_chunk` once that queue is empty. Memory is updated inside
        `predict_action_chunk`, so with `n_action_steps > 1` the recurrent state would
        advance once per chunk while training advanced it once per env step. Rather than
        silently produce out-of-distribution memory, fail loudly.
        """
        if self.config.effective_receding_horizon and self.config.n_action_steps != 1:
            raise ValueError(
                f"receding_horizon is active (use_memory={self.config.use_memory}, "
                f"receding_horizon={self.config.receding_horizon}) but n_action_steps="
                f"{self.config.n_action_steps}. Recurrent memory must advance once per env "
                "step to match training, so rollouts require n_action_steps=1. Set "
                "n_action_steps=1, or pass receding_horizon=False to accept the mismatch."
            )
        return super().select_action(batch, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()

        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

        mem_state = None
        if self.uses_memory:
            if self._mem_state is None or self._mem_state.shape[0] != tokens.shape[0]:
                self._mem_state = self.get_initial_memory(tokens.shape[0]).to(tokens.device)
            mem_state = self._mem_state

        actions, new_mem_state = self.model.sample_actions_mem(
            images, img_masks, tokens, masks, mem_state=mem_state, **kwargs
        )
        if new_mem_state is not None:
            # Must match the training-time update rule (see train.py, TBPTT/EMA branch)
            # and mu-VLA's eval loop (experiments/ablations/_common/episode_runner.py):
            # under "ema" the carried state is a blend, not the raw read-out.
            if self.config.memory_update == "ema":
                self._mem_state = MemoryModule.ema_update(
                    mem_state, new_mem_state, self.config.ema_alpha
                )
            else:
                self._mem_state = new_mem_state.detach()

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]
