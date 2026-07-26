"""
probe_embed_scale.py

Decide `PI05MemConfig.memory_write_scale`.

MEM tokens are read off the *output* of the PaliGemma backbone (after the final
RMSNorm) and fed back in as *input* embeddings on the next step. In mu-VLA's Llama
backbone those two spaces have comparable magnitude, so the state is fed back
verbatim. Gemma is not obviously the same: lerobot's `pi_gemma.py` drops the
`sqrt(hidden_size)` input-embedding multiplier, so the scales must be measured
rather than assumed.

This script reports the per-element std of:
  - image embeddings (`embed_image`),
  - language embeddings (`embed_language_tokens`),
  - the prefix output (what the MEM readout would be),
and prints the ratio that would align the readout with the input-embedding scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pi05_mem.episodic_dataset import EpisodicDatasetConfig, EpisodicLeRobotDataset
from pi05_mem.factory import make_config, make_policy, make_processors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "remember_color_3_vla_v0"


def describe(name: str, tensor: torch.Tensor) -> float:
    t = tensor.detach().to(torch.float32)
    std = t.std().item()
    print(f"{name:24s} shape={tuple(t.shape)} mean={t.mean().item():+.4f} std={std:.4f} "
          f"absmax={t.abs().max().item():.3f}")
    return std


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pretrained", default="lerobot/pi05_base")
    args = parser.parse_args()

    config = make_config(args.data, chunk_size=5, n_action_steps=1, use_memory=True, num_mem_tokens=4)
    policy = make_policy(config, pretrained=args.pretrained)
    preprocessor, _ = make_processors(config, args.data)
    policy.eval()

    ds = EpisodicLeRobotDataset(
        EpisodicDatasetConfig(roots=args.data, batch_size=args.batch_size, action_horizon=config.chunk_size)
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    batch = preprocessor(next(iter(loader)))

    model = policy.model
    images, img_masks = policy._preprocess_images(batch)  # noqa: SLF001
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]

    with torch.no_grad():
        img_emb = model.paligemma_with_expert.embed_image(images[0])
        lang_emb = model.paligemma_with_expert.embed_language_tokens(tokens)

        img_std = describe("embed_image", img_emb)
        lang_std = describe("embed_language_tokens", lang_emb)

        mem_state = model.get_initial_memory(tokens.shape[0]).to(tokens.device)
        describe("initial_memory", mem_state)

        actions = torch.zeros(
            tokens.shape[0], config.chunk_size, config.max_action_dim, device=tokens.device
        )
        noise = model.sample_noise(actions.shape, actions.device)
        time = model.sample_time(actions.shape[0], actions.device)
        _, new_mem = model.forward(images, img_masks, tokens, masks, actions, noise, time, mem_state)
        mem_std = describe("prefix_out[MEM]", new_mem)

    input_std = 0.5 * (img_std + lang_std)
    print()
    print(f"mean input-embedding std          = {input_std:.4f}")
    print(f"MEM readout std                   = {mem_std:.4f}")
    print(f"suggested memory_write_scale      = {input_std / mem_std:.4f}")
    print(f"1/sqrt(width) for reference       = {1.0 / (img_emb.shape[-1] ** 0.5):.4f}")
    print("PROBE_EMBED_SCALE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
