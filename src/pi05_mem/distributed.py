"""
distributed.py

Multi-GPU plumbing for the pi05-mem training loop, kept separate from `train.py` so
that the claims it makes - every rank trains on different data, every rank starts from
the same weights, gradients are averaged exactly once per optimizer step - can be
tested on their own.

Launch with `torchrun --standalone --nnodes 1 --nproc-per-node N`.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# How many of the first batches every rank fingerprints and cross-checks against the
# other ranks before training is allowed to proceed. MIKASA episodes are ~15 frames, so
# the window has to outlast one episode: a bug that only appears when a stream rolls
# over into its next episode would otherwise never be probed.
DDP_PROBE_STEPS = 32
# How many batches are recorded per rank for post-hoc auditing of the data split.
DDP_AUDIT_STEPS = 200
# Fraction of frames two ranks may legitimately share. Episodes are drawn with
# replacement, so a small overlap is expected; half the window is not.
MAX_ALLOWED_OVERLAP = 0.5
# NCCL's watchdog defaults to 10 minutes, and rank 0 writes a 9.4 GB checkpoint to NFS
# while the other seven sit in a barrier. One slow write must not kill the job.
DEFAULT_TIMEOUT_MINUTES = 60


@dataclass(frozen=True)
class DistInfo:
    """Where this process sits in the job. `world_size == 1` means single-process."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def setup_distributed(
    device: str = "cuda", timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES
) -> DistInfo:
    """Join the process group if launched under torchrun; otherwise stay single-process.

    `torch.cuda.set_device(local_rank)` is what makes a bare "cuda" resolve to this
    rank's GPU, so the rest of the code never has to spell out a device index.

    A launcher that sets LOCAL_RANK or WORLD_SIZE but not RANK is a hard error rather
    than a silent fallback to single-process: every process would believe it is rank 0
    and `is_main`, and all of them would write the same checkpoint at the same time.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if "RANK" not in os.environ:
        if world_size > 1 or "LOCAL_RANK" in os.environ:
            raise RuntimeError(
                f"launched with WORLD_SIZE={world_size} / LOCAL_RANK="
                f"{os.environ.get('LOCAL_RANK')} but without RANK. Every process would "
                "run as rank 0 and write the same checkpoint. Launch with torchrun."
            )
        logger.info("distributed: single process")
        return DistInfo()
    if world_size <= 1:
        logger.info("distributed: single process (WORLD_SIZE=1)")
        return DistInfo()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
    logger.info(
        "distributed: rank %d/%d (local_rank %d) backend=%s timeout=%dmin",
        rank, world_size, local_rank, backend, timeout_minutes,
    )
    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_module(module: torch.nn.Module, dist_info: DistInfo) -> int:
    """Give every rank rank 0's weights, then check that it took.

    Averaging gradients is only equivalent to one large batch if the ranks start from
    the same point. `DistributedDataParallel` broadcasts module state at construction;
    averaging by hand means doing this by hand too. The alternative - trusting that
    every rank ran the same seed through the same constructor - is an invariant nobody
    can see break: eight different models being averaged still produce a plausible
    loss curve.
    """
    if not dist_info.enabled:
        return 0
    tensors = [p.data for p in module.parameters()] + [b.data for b in module.buffers()]
    with torch.no_grad():
        for tensor in tensors:
            dist.broadcast(tensor, src=0)
    check_module_identical(module, dist_info)
    logger.info("broadcast %d tensors from rank 0; all ranks agree", len(tensors))
    return len(tensors)


def check_module_identical(module: torch.nn.Module, dist_info: DistInfo) -> float:
    """Assert every rank holds the same parameters. Returns the largest disagreement."""
    if not dist_info.enabled:
        return 0.0
    checksums = torch.stack([p.detach().double().sum() for p in module.parameters()])
    high, low = checksums.clone(), checksums.clone()
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    gap = (high - low).abs()
    spread = float(gap.max().item())
    if spread > 0.0:
        raise RuntimeError(
            f"ranks hold different weights (parameter #{int(gap.argmax().item())} differs "
            f"by {spread:.3e}). Averaged gradients would be applied to different models, "
            "which trains world_size different policies and reports one loss."
        )
    return spread


class GradientSynchronizer:
    """Average gradients across ranks, once, right before the optimizer step.

    `torch.nn.parallel.DistributedDataParallel` is deliberately not used here. Its
    reducer assumes one backward per forward; TBPTT does K forwards and a single
    backward over their summed loss, and gradient accumulation adds further backwards
    per optimizer step. Under that schedule the autograd hooks fire at the wrong
    times. Averaging explicitly at the single point where gradients are consumed is
    both correct and trivially auditable.

    Parameters are bucketed into flat buffers to avoid one collective per tensor
    (pi0.5 has ~1600 of them). A parameter whose grad is None on this rank - which
    happens for `initial_memory` in a window with no episode start - contributes zeros,
    exactly as mu-VLA's explicit `all_reduce` of that grad did. A parameter whose grad
    is None on *every* rank keeps its None, so AdamW's decoupled weight decay does not
    quietly shrink a parameter the model never used.
    """

    def __init__(self, parameters, world_size: int, bucket_elems: int = 64_000_000):
        if isinstance(parameters, torch.nn.Module):
            self._source = parameters.parameters
        else:
            frozen = list(parameters)
            self._source = lambda: frozen
        self.params = [p for p in self._source() if p.requires_grad]
        self._param_ids = {id(p) for p in self.params}
        self._verified = False
        self.world_size = world_size
        self.buckets: list[list[torch.nn.Parameter]] = []
        current: list[torch.nn.Parameter] = []
        current_elems = 0
        current_dtype = None
        for param in self.params:
            n = param.numel()
            if current and (param.dtype != current_dtype or current_elems + n > bucket_elems):
                self.buckets.append(current)
                current, current_elems = [], 0
            if not current:
                current_dtype = param.dtype
            current.append(param)
            current_elems += n
        if current:
            self.buckets.append(current)

    def _verify_coverage(self) -> None:
        """Everything carrying a gradient must be something this synchronizer averages.

        The optimizer and `clip_grad_norm_` walk `model.parameters()` afresh every step,
        while the bucket list is built once. A parameter created or unfrozen later would
        be stepped with this rank's own gradient instead of the average, and nothing
        downstream would notice.
        """
        self._verified = True
        stray = [p for p in self._source() if p.grad is not None and id(p) not in self._param_ids]
        if stray:
            raise RuntimeError(
                f"{len(stray)} parameter(s) carry gradients but are not registered with "
                "GradientSynchronizer (shapes: "
                + ", ".join(str(tuple(p.shape)) for p in stray[:3])
                + "). They would be updated with per-rank, unaveraged gradients. "
                "Rebuild the synchronizer after changing requires_grad."
            )

    def sync(self) -> None:
        if self.world_size <= 1:
            return
        if not self._verified:
            self._verify_coverage()
        # One collective for "did anybody use this parameter", so a parameter that is
        # unused on every rank can be left at grad=None instead of a materialised zero.
        device = self.params[0].device if self.params else torch.device("cpu")
        presence = torch.tensor(
            [1 if p.grad is not None else 0 for p in self.params], dtype=torch.int32, device=device
        )
        dist.all_reduce(presence, op=dist.ReduceOp.MAX)
        used_anywhere = presence.tolist()

        index = 0
        for bucket in self.buckets:
            flat = torch.cat(
                [(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in bucket]
            )
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat /= self.world_size
            offset = 0
            for p in bucket:
                n = p.numel()
                if not used_anywhere[index]:
                    p.grad = None
                elif p.grad is None:
                    p.grad = flat[offset : offset + n].view_as(p).clone()
                else:
                    p.grad.copy_(flat[offset : offset + n].view_as(p))
                offset += n
                index += 1


def fingerprint_batch(raw_batch: dict) -> list[str]:
    """One string per stream: which dataset / episode / frame that slot is on.

    This is the identity of the data a rank is training on, so comparing it across
    ranks is a direct check that the GPUs are not all replaying the same batch.
    """
    names = list(raw_batch["dataset_name"])
    episodes = [int(v) for v in raw_batch["episode_index"]]
    frames = [int(v) for v in raw_batch["frame_index"]]
    return [f"{n}:{e}:{f}" for n, e, f in zip(names, episodes, frames)]


def check_ranks_see_different_data(fingerprints: list[str], dist_info: DistInfo) -> dict:
    """Refuse to train if two ranks are walking the same trajectories.

    Every rank passes the fingerprints of its first `DDP_PROBE_STEPS` batches. They are
    gathered and compared as *multisets*, not as ordered lists: a bug that permutes the
    stream-to-slot assignment per rank while handing every rank the same trajectories
    passes an ordered comparison and still makes an N-GPU job N times redundant.

    Partial overlap is legitimate - episodes are drawn with replacement, exactly as in
    mu-VLA, so two of 32 streams landing on one episode happens. Sharing half the
    window is not, which is where `MAX_ALLOWED_OVERLAP` draws the line.
    """
    if not dist_info.enabled:
        return {"world_size": 1, "checked": False}

    payload = "|".join(fingerprints)
    gathered: list[str | None] = [None] * dist_info.world_size
    dist.all_gather_object(gathered, payload)

    per_rank = [Counter(str(p).split("|")) for p in gathered]
    duplicates = []
    overlaps = []
    worst = (0.0, -1, -1)
    for i in range(dist_info.world_size):
        for j in range(i + 1, dist_info.world_size):
            shared = sum((per_rank[i] & per_rank[j]).values())
            fraction = shared / max(1, sum(per_rank[i].values()))
            overlaps.append(fraction)
            if fraction > worst[0]:
                worst = (fraction, i, j)
            if per_rank[i] == per_rank[j]:
                duplicates.append((i, j))
    report = {
        "world_size": dist_info.world_size,
        "checked": True,
        "probe_steps": DDP_PROBE_STEPS,
        "items_per_rank": len(fingerprints),
        "max_pairwise_overlap": round(max(overlaps), 4) if overlaps else 0.0,
        "mean_pairwise_overlap": round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0,
        "identical_rank_pairs": duplicates,
        "max_allowed_overlap": MAX_ALLOWED_OVERLAP,
    }
    if duplicates:
        raise RuntimeError(
            "DDP data check failed: ranks "
            + ", ".join(f"{i}=={j}" for i, j in duplicates)
            + " are walking the same frames. Every GPU would train on the same data, so "
            "the run costs N times more for nothing. Check that EpisodicDatasetConfig "
            "receives rank/world_size.\n" + json.dumps(report)
        )
    if report["max_pairwise_overlap"] > MAX_ALLOWED_OVERLAP:
        raise RuntimeError(
            f"DDP data check failed: ranks {worst[1]} and {worst[2]} share "
            f"{worst[0]:.0%} of their frames (limit {MAX_ALLOWED_OVERLAP:.0%}). That is "
            "far more than drawing episodes with replacement explains, so the rank "
            "offset is probably not reaching the stream seeds.\n" + json.dumps(report)
        )
    return report


def reduce_mean(value: float, dist_info: DistInfo, device: torch.device) -> float:
    """Average a scalar over ranks, for logging that describes the whole job."""
    if not dist_info.enabled:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / dist_info.world_size)
