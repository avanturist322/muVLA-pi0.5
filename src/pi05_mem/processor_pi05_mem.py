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
processor_pi05_mem.py

Same pre/post-processing pipeline as `lerobot.policies.pi05.processor_pi05`, with the
tokenizer path made configurable.

`make_pi05_pre_post_processors` hardcodes `tokenizer_name="google/paligemma-3b-pt-224"`,
which is a gated repo. `scripts/fetch_tokenizer.py` puts a byte-identical copy of the
PaliGemma SentencePiece tokenizer in `assets/paligemma_tokenizer/` (verified against the
canonical Big Vision file), and this factory points the pipeline at it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_pi05_mem import PI05MemConfig

DEFAULT_TOKENIZER_PATH = Path(__file__).resolve().parents[2] / "assets" / "paligemma_tokenizer"


def make_pi05_mem_pre_post_processors(
    config: PI05MemConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    tokenizer_name: str | Path | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    if tokenizer_name is None:
        tokenizer_name = DEFAULT_TOKENIZER_PATH
    tokenizer_name = str(tokenizer_name)

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        relative_step,
        # NormalizerProcessorStep must precede the state tokenizer: the latter expects
        # state already normalized to [-1, 1] before discretizing into 256 bins.
        steps.normalize,
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
    ]

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
