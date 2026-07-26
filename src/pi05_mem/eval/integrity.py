"""
integrity.py

Two checks that stand between "the eval ran" and "the eval measured the policy we
trained". Both exist because the failure they catch is silent: the numbers come out
plausible-looking and low, and low is exactly what one expects from a hard benchmark.

1. `verify_checkpoint_loaded` - LeRobot's `PI05Policy._load_as_safetensor` wraps the
   whole load in `except Exception: print("Warning: ...")` and reports missing keys to
   stdout (vendored copy, modeling_pi05.py:855-857). We call it with `strict=False`
   because the pretrained pi0.5 checkpoint has no memory module. Together that means a
   renamed tensor, a truncated shard or an outright failed read leaves those weights at
   their PaliGemma or random init and the eval proceeds.

2. `verify_eval_datasets` - pi0.5 discretizes the normalized state into 256 bins and
   writes them into the *text prompt*. Evaluating with quantiles pooled over a
   different set of datasets than training used does not merely rescale a tensor, it
   feeds the model a different prompt. The binding between checkpoint and dataset list
   is by convention (both come from the same training task list), so it is checked.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Parameters legitimately absent from a checkpoint's safetensors file.
ALLOWED_MISSING_PREFIXES = ("model.memory_module.",)
# How many tensors get their values compared, not just their names and shapes.
VALUE_SPOT_CHECKS = 16


def _remap_key(key: str) -> str:
    """The name-only half of `PI05Policy._fix_pytorch_state_dict_keys`, plus the prefix.

    Value-dependent branches in the original (lm_head duplication, adaRMS skips) do not
    apply to a checkpoint this repo wrote, and a stale key surviving them would show up
    here as an unexpected key rather than pass unnoticed.
    """
    if key.startswith("action_time_mlp_in."):
        key = key.replace("action_time_mlp_in.", "time_mlp_in.")
    elif key.startswith("action_time_mlp_out."):
        key = key.replace("action_time_mlp_out.", "time_mlp_out.")
    return key if key.startswith("model.") else f"model.{key}"


def verify_checkpoint_loaded(checkpoint: Path, policy) -> dict:
    """Raise unless every tensor in the checkpoint reached the live model.

    Names and shapes are read from the safetensors header, which costs no I/O beyond a
    few KB. A deterministic sample of tensors is then compared by value, because a
    header can agree while `load_state_dict` never ran at all.
    """
    from safetensors import safe_open

    path = Path(checkpoint) / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(
            f"{checkpoint} has no model.safetensors; nothing was loaded and the eval "
            "would score a randomly initialized pi0.5"
        )

    live = dict(policy.state_dict())
    with safe_open(path, framework="pt", device="cpu") as handle:
        file_keys = list(handle.keys())
        mismatched, unexpected = [], []
        for key in file_keys:
            target = _remap_key(key)
            if target not in live:
                unexpected.append(key)
                continue
            want = tuple(handle.get_slice(key).get_shape())
            got = tuple(live[target].shape)
            if want != got:
                mismatched.append(f"{key}: file {want} vs model {got}")

        missing = [
            name for name in live
            if name not in {_remap_key(k) for k in file_keys}
            and not name.startswith(ALLOWED_MISSING_PREFIXES)
        ]

        problems = []
        if unexpected:
            problems.append(f"{len(unexpected)} tensor(s) in the file have no home in the "
                            f"model, e.g. {unexpected[:3]}")
        if missing:
            problems.append(f"{len(missing)} model parameter(s) are absent from the file "
                            f"and kept their init, e.g. {missing[:3]}")
        if mismatched:
            problems.append(f"{len(mismatched)} shape mismatch(es): {mismatched[:3]}")
        if problems:
            raise RuntimeError(
                f"checkpoint {checkpoint} did not load cleanly into the eval policy - "
                + "; ".join(problems)
                + ". LeRobot loads with strict=False and only prints this, so the eval "
                "would have scored partly untrained weights."
            )

        # Values, on a stride-sampled subset: the header can agree while nothing loaded.
        stride = max(1, len(file_keys) // VALUE_SPOT_CHECKS)
        checked = 0
        for key in file_keys[::stride]:
            want = handle.get_tensor(key)
            got = live[_remap_key(key)].detach().to("cpu", dtype=want.dtype)
            if not torch.equal(want, got):
                raise RuntimeError(
                    f"checkpoint {checkpoint}: tensor '{key}' in the model differs from "
                    "the file. The weights on the GPU are not the weights that were "
                    "trained - the load silently failed."
                )
            checked += 1

    logger.info(
        "checkpoint verified: %d tensors match by name and shape, %d by value",
        len(file_keys), checked,
    )
    return {"tensors": len(file_keys), "value_checked": checked}


def verify_eval_datasets(checkpoint: Path, datasets: list[Path]) -> dict:
    """Raise if the eval dataset list differs from the one the checkpoint trained on.

    `train_config.json` sits at the *run* root, one level above a `final/` or `step-N/`
    checkpoint directory, so both are searched. A checkpoint from elsewhere has no such
    file; that is reported as unverified rather than treated as agreement, because the
    honest statement is "we could not check", not "it is fine".
    """
    names = sorted(Path(d).name for d in datasets)
    checkpoint = Path(checkpoint)
    for candidate in (checkpoint / "train_config.json", checkpoint.parent / "train_config.json"):
        if not candidate.exists():
            continue
        saved = json.loads(candidate.read_text())
        trained_on = saved.get("datasets") or saved.get("data") or []
        if isinstance(trained_on, str):
            trained_on = [t for t in trained_on.split(",") if t]
        trained = sorted(Path(str(t)).name for t in trained_on)
        if not trained:
            break
        if trained != names:
            raise RuntimeError(
                f"eval would use datasets {names} but {candidate} says the checkpoint "
                f"was trained on {trained}. pi0.5 writes the quantile-normalized state "
                "into the text prompt as 256 discrete bins, so a different pooled "
                "mixture means a different prompt, not a different scale. Pass the "
                "training mixture, or re-run with --data matching it."
            )
        logger.info("dataset mixture matches training: %s", ", ".join(names))
        return {"verified": True, "datasets": names, "source": str(candidate)}

    logger.warning(
        "no train_config.json next to %s: cannot confirm that %s is the mixture whose "
        "pooled quantiles this checkpoint was trained with",
        checkpoint, ", ".join(names),
    )
    return {"verified": False, "datasets": names, "source": None}
