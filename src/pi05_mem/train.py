"""
train.py

Training loop for pi05-mem, ported from mu-VLA's `vla-scripts/finetune.py`
(the TBPTT / EMA section, lines ~1270-1400).

Why a custom loop rather than `lerobot-train`: recurrent memory requires the batch to
stay time-coherent across optimizer steps and the graph to span several env steps
(TBPTT). Neither fits LeRobot's shuffled-sampler training loop.

Two memory update rules, selected by `--memory-update`:

  tbptt  Keep the autograd graph across K = --tbptt-length env steps, do one backward
         over the accumulated loss, then detach the memory at the window boundary.
  ema    Backward every step; the next step's memory input is
         alpha * M_out + (1 - alpha) * M_in, with both operands detached, so no
         gradient crosses step boundaries.

Episode starts (`is_first`) always reset the memory to the learnable `initial_memory`,
with gradient flowing into it.

Multi-GPU: launch with `torchrun --standalone --nnodes 1 --nproc-per-node N`. Every
rank runs its own B time-coherent streams over *different* episodes (the stream seed
carries the rank), and gradients are averaged manually right before the optimizer step
- see `GradientSynchronizer` for why torch's DDP wrapper is not used. The loop refuses
to train if two ranks turn out to be walking the same trajectories; see
`check_ranks_see_different_data`.

No wandb: metrics go to stdout and to `<output>/metrics.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .distributed import (
    DDP_AUDIT_STEPS,
    DDP_PROBE_STEPS,
    DistInfo,
    GradientSynchronizer,
    check_ranks_see_different_data,
    cleanup_distributed,
    fingerprint_batch,
    broadcast_module,
    reduce_mean,
    setup_distributed,
)
from .episodic_dataset import EpisodicDatasetConfig, EpisodicLeRobotDataset, normalize_roots
from .factory import make_config, make_policy, make_processors
from .memory import MemoryModule
from .memory_meta import save_memory_module

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TrainConfig:
    """Everything the loop needs. Memory flag names mirror mu-VLA."""

    # One dataset directory, or several for multi-task training (mu-VLA's MIKASA_ENVS).
    data: tuple[Path, ...] = ()
    output: Path = Path("runs/unnamed")
    pretrained: str | None = "lerobot/pi05_base"
    # Per-dataset sampling probability; None = uniform over datasets, as in mu-VLA.
    dataset_weights: tuple[float, ...] | None = None

    batch_size: int = 8
    action_horizon: int = 5
    max_steps: int = 100
    grad_accumulation_steps: int = 1
    learning_rate: float = 2.5e-5
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = False
    max_episode_steps: int | None = None

    # optimizer / schedule (defaults follow LeRobot's pi0.5 preset)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 1000
    lr_min_ratio: float = 0.1
    num_steps_before_decay: int = 100_000

    # mu-VLA memory flags
    use_memory: bool = False
    num_mem_tokens: int = 4
    memory_update: str = "tbptt"
    ema_alpha: float = 0.1
    tbptt_length: int = 2
    attention_mask_mode: str = "custom"
    memory_write_scale: float = 5.8
    memory_init_std: float = 4.0
    memory_log_freq: int = 0
    memory_expensive_log_freq: int = 0

    log_freq: int = 1
    save_freq: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", normalize_roots(self.data))
        object.__setattr__(self, "output", Path(self.output))
        if self.dataset_weights is not None:
            object.__setattr__(self, "dataset_weights", tuple(float(w) for w in self.dataset_weights))

    @property
    def dataset_names(self) -> list[str]:
        return [path.name for path in self.data]


MEMORY_FLAGS = (
    "use_memory",
    "num_mem_tokens",
    "memory_update",
    "ema_alpha",
    "tbptt_length",
    "attention_mask_mode",
    "memory_write_scale",
    "memory_init_std",
    "memory_log_freq",
    "memory_expensive_log_freq",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train pi0.5 with mu-VLA recurrent memory")
    p.add_argument(
        "--data",
        required=True,
        help=(
            "LeRobot v3 dataset directory, or a comma-separated list of them for "
            "multi-task training (e.g. 'data/remember_color_3_vla_v0,data/shell_game_push_vla_v0')"
        ),
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="prefix for --data entries, so they can be bare dataset names (mu-VLA's MIKASA_ENVS style)",
    )
    p.add_argument(
        "--dataset-weights",
        default=None,
        help="comma-separated sampling weights, one per --data entry; default is uniform",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pretrained", default="lerobot/pi05_base", help="'none' to train from scratch")

    p.add_argument(
        "--batch-size", type=int, default=8, help="streams PER RANK; global batch = B * world_size"
    )
    p.add_argument("--action-horizon", type=int, default=5, help="== policy chunk_size")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--grad-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2.5e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-episode-steps", type=int, default=None)

    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-schedule", choices=("cosine", "multistep", "constant"), default="cosine")
    p.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=1000,
        help="warmup length in OPTIMIZER steps (not dataloader steps)",
    )
    p.add_argument(
        "--lr-min-ratio", type=float, default=0.1, help="cosine floor, as a fraction of peak LR"
    )
    p.add_argument("--num-steps-before-decay", type=int, default=100_000, help="multistep only")

    p.add_argument("--use-memory", action="store_true")
    p.add_argument("--num-mem-tokens", type=int, default=4)
    p.add_argument("--memory-update", choices=("tbptt", "ema"), default="tbptt")
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--tbptt-length", type=int, default=2, help="K")
    p.add_argument("--attention-mask-mode", choices=("custom", "full"), default="custom")
    p.add_argument("--memory-write-scale", type=float, default=5.8)
    p.add_argument("--memory-init-std", type=float, default=4.0)
    p.add_argument("--memory-log-freq", type=int, default=0)
    p.add_argument("--memory-expensive-log-freq", type=int, default=0)

    p.add_argument("--log-freq", type=int, default=1)
    p.add_argument("--save-freq", type=int, default=0)
    return p


def parse_data_arg(data: str, data_root: Path | None = None) -> tuple[Path, ...]:
    """'a,b' (+ optional --data-root prefix) -> (Path('root/a'), Path('root/b'))."""
    names = [part.strip() for part in str(data).split(",") if part.strip()]
    if not names:
        raise ValueError("--data is empty")
    paths = [Path(name) if data_root is None else Path(data_root) / name for name in names]
    if len(set(paths)) != len(paths):
        raise ValueError(f"--data lists the same dataset twice: {[str(p) for p in paths]}")
    return tuple(paths)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    kwargs = vars(args).copy()
    if kwargs.get("pretrained") in ("none", "None", ""):
        kwargs["pretrained"] = None
    kwargs["data"] = parse_data_arg(kwargs["data"], kwargs.pop("data_root", None))
    weights = kwargs.get("dataset_weights")
    if isinstance(weights, str):
        kwargs["dataset_weights"] = tuple(float(w) for w in weights.split(",") if w.strip())
    return TrainConfig(**kwargs)


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- learning-rate schedule ----------------------------------------------------


def build_lr_lambda(cfg: TrainConfig, total_optimizer_steps: int):
    """LR multiplier as a function of the optimizer-step index.

    `cosine` reproduces mu-VLA's `_cosine_with_warmup` verbatim: a linear warmup from
    0.1 to 1.0 over `lr_warmup_steps`, then a cosine from 1.0 down to `lr_min_ratio`
    over the remaining steps. Config 4 of the mu-VLA experiments used exactly this,
    and the baseline config has to use it too for the comparison to isolate memory.
    """
    if cfg.lr_schedule == "constant":
        return lambda step: 1.0

    if cfg.lr_schedule == "multistep":
        milestone = cfg.num_steps_before_decay
        return lambda step: 1.0 if step < milestone else 0.1

    warmup = cfg.lr_warmup_steps
    min_ratio = cfg.lr_min_ratio
    total = max(total_optimizer_steps, 1)

    def _cosine_with_warmup(step: int) -> float:
        if warmup > 0 and step < warmup:
            return 0.1 + 0.9 * (step / warmup)
        decay_steps = max(total - warmup, 1)
        progress = min((step - warmup) / decay_steps, 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return _cosine_with_warmup


def _all_reduce_initial_memory_grad(memory_module: MemoryModule) -> None:
    """DDP parity with mu-VLA: `initial_memory` is only touched on `is_first` steps,
    so its gradient can be absent on some ranks and must be averaged explicitly.

    Kept for tests and for single-purpose use; the training loop no longer calls it,
    because `GradientSynchronizer` already averages every gradient (materializing
    zeros for the ranks where this one is absent) at the optimizer step.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    grad = memory_module.initial_memory.grad
    if grad is None:
        grad = torch.zeros_like(memory_module.initial_memory)
        memory_module.initial_memory.grad = grad
    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    grad /= dist.get_world_size()


def _memory_diagnostics(mem_in: torch.Tensor, mem_out: torch.Tensor, expensive: bool) -> dict:
    diag = {
        "mem_in_norm": mem_in.detach().float().norm(dim=-1).mean().item(),
        "mem_out_norm": mem_out.detach().float().norm(dim=-1).mean().item(),
    }
    if expensive:
        delta = (mem_out.detach().float() - mem_in.detach().float()).norm(dim=-1)
        diag["mem_drift_mean"] = delta.mean().item()
        diag["mem_drift_max"] = delta.max().item()
        diag["mem_cos"] = (
            torch.nn.functional.cosine_similarity(
                mem_in.detach().float(), mem_out.detach().float(), dim=-1
            )
            .mean()
            .item()
        )
    return diag


def _jsonable(value):
    """Paths and tuples of Paths -> strings, for the config dump."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def train(cfg: TrainConfig, dist_info: DistInfo | None = None) -> dict:
    dist_info = dist_info if dist_info is not None else DistInfo()

    # Identical seed everywhere: the policy (including the randomly initialized
    # `initial_memory`) must start from the same weights on every rank, since nothing
    # broadcasts them afterwards. Data diversity comes from the rank-dependent stream
    # seed inside EpisodicDatasetConfig, not from this one.
    set_seed(cfg.seed)

    if dist_info.is_main:
        cfg.output.mkdir(parents=True, exist_ok=True)
    if dist_info.enabled:
        dist.barrier()
    audit_dir = cfg.output / "ddp"
    audit_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = cfg.output / (
        "metrics.jsonl" if dist_info.is_main else f"metrics-rank{dist_info.rank}.jsonl"
    )
    if dist_info.is_main:
        (cfg.output / "train_config.json").write_text(
            json.dumps(
                {
                    **{k: _jsonable(v) for k, v in asdict(cfg).items()},
                    "world_size": dist_info.world_size,
                    "global_batch_size": cfg.batch_size * dist_info.world_size,
                },
                indent=2,
            )
        )

    memory_kwargs = {flag: getattr(cfg, flag) for flag in MEMORY_FLAGS}
    policy_config = make_config(
        cfg.data,
        chunk_size=cfg.action_horizon,
        n_action_steps=1,
        device=cfg.device,
        dtype=cfg.dtype,
        **memory_kwargs,
    )
    policy_config.gradient_checkpointing = cfg.gradient_checkpointing
    policy_config.optimizer_lr = cfg.learning_rate

    policy = make_policy(policy_config, pretrained=cfg.pretrained)
    preprocessor, _ = make_processors(policy_config, cfg.data)
    policy.train()

    # Averaging gradients by hand is only equivalent to one large batch if every
    # rank starts from the same weights. DistributedDataParallel broadcasts module
    # state at construction; this loop does not use it, so it broadcasts here.
    broadcast_module(policy, dist_info)

    # Weights are now identical across ranks; from here on the per-step randomness
    # (flow-matching noise and timesteps) should differ, or each rank would draw the
    # same noise for its own - different - batch.
    torch.manual_seed(cfg.seed + 10_007 * dist_info.rank)
    torch.cuda.manual_seed_all(cfg.seed + 10_007 * dist_info.rank)

    dataset = EpisodicLeRobotDataset(
        EpisodicDatasetConfig(
            roots=cfg.data,
            batch_size=cfg.batch_size,
            action_horizon=cfg.action_horizon,
            seed=cfg.seed,
            rank=dist_info.rank,
            world_size=dist_info.world_size,
            max_episode_steps=cfg.max_episode_steps,
            dataset_weights=cfg.dataset_weights,
        )
    )
    # num_workers=0 and shuffle=False are required: the round-robin stream ordering is
    # what makes batch[i] time-coherent.
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0, shuffle=False)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    # The module, not a snapshot of its parameters: sync() then checks on its first
    # call that nothing acquired a gradient outside the buckets it averages.
    synchronizer = GradientSynchronizer(policy, dist_info.world_size)

    model = policy.model
    memory_module = model.memory_module
    effective_tbptt = policy_config.effective_tbptt_length
    effective_accum_steps = effective_tbptt * cfg.grad_accumulation_steps
    total_optimizer_steps = max(cfg.max_steps // effective_accum_steps, 1)
    scheduler = LambdaLR(optimizer, lr_lambda=build_lr_lambda(cfg, total_optimizer_steps))

    logger.info(
        "training on %d dataset(s): %s",
        len(dataset.shards),
        ", ".join(
            f"{name} ({episodes} ep)" for name, episodes in dataset.episodes_per_dataset().items()
        ),
    )
    logger.info(
        "training: use_memory=%s update=%s K=%d mask=%s | B=%d/rank world=%d global_B=%d "
        "steps=%d accum=%d optimizer_steps=%d | lr=%.2e schedule=%s warmup=%d min_ratio=%.2f",
        cfg.use_memory,
        cfg.memory_update,
        effective_tbptt,
        cfg.attention_mask_mode,
        cfg.batch_size,
        dist_info.world_size,
        cfg.batch_size * dist_info.world_size,
        cfg.max_steps,
        cfg.grad_accumulation_steps,
        total_optimizer_steps,
        cfg.learning_rate,
        cfg.lr_schedule,
        cfg.lr_warmup_steps,
        cfg.lr_min_ratio,
    )

    device = torch.device(cfg.device)
    mem_state: torch.Tensor | None = None
    tbptt_count = 0
    tbptt_loss_accum: torch.Tensor | float = 0.0
    backward_count = 0
    optimizer_steps = 0
    history: list[dict] = []
    multi_task = len(dataset.shards) > 1
    dataset_mix: Counter[str] = Counter()
    probe_fingerprints: list[str] = []
    ddp_report: dict = {"world_size": dist_info.world_size, "checked": False}
    start = time.time()

    metrics_file = metrics_path.open("w")
    audit_file = (audit_dir / f"rank{dist_info.rank}-batches.jsonl").open("w")
    try:
        for step, raw_batch in enumerate(loader):
            if step >= cfg.max_steps:
                break

            fingerprints = fingerprint_batch(raw_batch)
            if step < DDP_AUDIT_STEPS:
                audit_file.write(
                    json.dumps({"step": step, "rank": dist_info.rank, "slots": fingerprints}) + "\n"
                )
                audit_file.flush()
            if step < DDP_PROBE_STEPS:
                probe_fingerprints.extend(fingerprints)
            if step == DDP_PROBE_STEPS - 1:
                # Before any real compute is spent: prove the ranks are not clones.
                ddp_report = check_ranks_see_different_data(probe_fingerprints, dist_info)
                if dist_info.is_main and ddp_report.get("checked"):
                    (cfg.output / "ddp_check.json").write_text(json.dumps(ddp_report, indent=2))
                    logger.info("DDP data check passed: %s", json.dumps(ddp_report))

            is_first = raw_batch["is_first"].to(device)
            batch = preprocessor(raw_batch)

            mem_in = None
            if cfg.use_memory:
                if mem_state is None:
                    mem_state = memory_module.get_initial_state(cfg.batch_size).to(device)
                if cfg.memory_update == "tbptt":
                    # Keep everything in the graph inside the TBPTT window; the window
                    # boundary below is the only place we detach.
                    no_detach = torch.zeros_like(is_first)
                    mem_state = memory_module.reset_episodes(mem_state, is_first, no_detach)
                else:
                    # EMA: continuing episodes get a detached state.
                    mem_state = memory_module.reset_episodes(mem_state, is_first)
                mem_in = mem_state

            loss, loss_dict, new_mem_state = policy.forward_with_memory(batch, mem_state=mem_state)

            record = {"step": step, "loss": loss_dict["loss"], "is_first": int(is_first.sum().item())}
            if multi_task:
                names = list(raw_batch["dataset_name"])
                dataset_mix.update(names)
                record["datasets"] = {name: names.count(name) for name in sorted(set(names))}

            if cfg.use_memory and cfg.memory_update == "tbptt":
                tbptt_loss_accum = tbptt_loss_accum + loss
                tbptt_count += 1
                mem_state = new_mem_state  # stays in the graph
                if tbptt_count >= effective_tbptt:
                    (tbptt_loss_accum / effective_accum_steps).backward()
                    backward_count += 1
                    mem_state = mem_state.detach()
                    tbptt_count = 0
                    tbptt_loss_accum = 0.0
            else:
                (loss / effective_accum_steps).backward()
                backward_count += 1
                if cfg.use_memory:
                    mem_state = MemoryModule.ema_update(mem_in, new_mem_state, cfg.ema_alpha)

            if cfg.use_memory and cfg.memory_log_freq and step % cfg.memory_log_freq == 0:
                expensive = bool(
                    cfg.memory_expensive_log_freq and step % cfg.memory_expensive_log_freq == 0
                )
                record.update(_memory_diagnostics(mem_in, new_mem_state, expensive))

            if backward_count and backward_count % cfg.grad_accumulation_steps == 0:
                # One collective for the whole model, at the only point where the
                # gradients are read. All ranks reach this together: `backward_count`
                # follows the same schedule everywhere.
                synchronizer.sync()
                if cfg.use_memory:
                    # Read here rather than in the memory-diagnostics block above. That
                    # block samples on a fixed `--memory-log-freq` modulus, but under
                    # TBPTT the backward only fires every K microsteps, so with K=8 and
                    # freq=100 the two schedules never coincide: the sampled steps land
                    # on `step % 8 in {0, 4}`, the backward on `step % 8 == 7`. Every one
                    # of the 242 logged rows of config-b-mem therefore read a grad that
                    # `zero_grad(set_to_none=True)` had just cleared, and reported 0.0
                    # for a parameter that demonstrably was training - comparing
                    # step-012000 with step-024000 gives |dx| = 0.468 on `initial_memory`.
                    # A metric whose only possible value is "frozen" cannot report
                    # freezing. Before `clip_grad_norm_`, which rescales in place.
                    grad = memory_module.initial_memory.grad
                    record["initial_memory_grad_norm"] = (
                        0.0 if grad is None else grad.norm().item()
                    )
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
                backward_count = 0
                record["grad_norm"] = float(grad_norm)
                record["optimizer_step"] = True
                record["lr"] = optimizer.param_groups[0]["lr"]

            record["elapsed_s"] = round(time.time() - start, 2)
            history.append(record)
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

            if cfg.log_freq and step % cfg.log_freq == 0:
                # A job-wide loss, not this rank's slice of it.
                mean_loss = reduce_mean(record["loss"], dist_info, device)
                record["loss_global"] = mean_loss
                if dist_info.is_main:
                    logger.info(
                        "step %4d | loss %.4f | new_eps %d%s%s",
                        step,
                        mean_loss,
                        record["is_first"],
                        f" | grad {record['grad_norm']:.3f}" if "grad_norm" in record else "",
                        f" | lr {record['lr']:.2e}" if "lr" in record else "",
                    )

            if cfg.save_freq and step > 0 and step % cfg.save_freq == 0:
                if dist_info.is_main:
                    _save(policy, policy_config, cfg.output / f"step-{step:06d}")
                if dist_info.enabled:
                    dist.barrier()
    finally:
        metrics_file.close()
        audit_file.close()

    if dist_info.is_main:
        _save(policy, policy_config, cfg.output / "final")
    if dist_info.enabled:
        dist.barrier()

    losses = [r["loss"] for r in history]
    summary = {
        "steps": len(history),
        "optimizer_steps": optimizer_steps,
        "world_size": dist_info.world_size,
        "global_batch_size": cfg.batch_size * dist_info.world_size,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "mean_first_10": sum(losses[:10]) / max(1, len(losses[:10])),
        "mean_last_10": sum(losses[-10:]) / max(1, len(losses[-10:])),
        "wall_time_s": round(time.time() - start, 1),
        "datasets": dataset.dataset_names,
        "ddp_check": ddp_report,
    }
    if multi_task:
        # Frames seen per dataset: the realized mixture, which for uniform sampling is
        # proportional to episodes drawn x episode length, not to dataset size.
        summary["dataset_mix"] = dict(sorted(dataset_mix.items()))
    if dist_info.is_main:
        (cfg.output / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("summary: %s", summary)
    return summary


def _save(policy, policy_config, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(directory)
    save_memory_module(directory, policy.model, policy_config)
    logger.info("saved checkpoint to %s", directory)


def main() -> None:
    cfg = config_from_args(build_parser().parse_args())
    dist_info = setup_distributed(cfg.device)
    # force=True: importing lerobot/transformers already installs a root handler, which
    # would make a plain basicConfig() a no-op and swallow every progress line.
    logging.basicConfig(
        level=logging.INFO if dist_info.is_main else logging.WARNING,
        format=f"%(asctime)s %(levelname)s rank{dist_info.rank} %(name)s: %(message)s",
        force=True,
    )
    try:
        train(cfg, dist_info)
    except BaseException:
        # One rank raising leaves the others blocked in the next collective until
        # the watchdog fires, which buries this traceback under seven timeouts.
        # Exit hard instead: torchrun sees a dead child and tears the group down.
        logger.exception("rank %d failed", dist_info.rank)
        for handler in logging.getLogger().handlers:
            handler.flush()
        if dist_info.enabled:
            os._exit(1)
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
