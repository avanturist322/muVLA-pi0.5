"""
memory.py

MemoryModule for pi05-mem: learnable recurrent memory tokens injected into the
multimodal prefix of the transformer sequence.

Ported from mu-VLA (`prismatic/models/memory.py`, OpenVLA-OFT fork) with the
semantics preserved exactly, so that ablations and hyperparameters transfer
one-to-one between the two code bases.

At t=0 (episode start): memory is initialized from a learnable nn.Parameter.
At t>0: memory is the hidden state read from MEM positions at the previous step.
TBPTT: continuing episodes get detached memory; new episodes keep gradient flow
through initial_memory so the model learns to initialize memory correctly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """Learnable recurrent memory tokens.

    Args:
        num_mem_tokens: Number of MEM tokens injected into the prefix.
        hidden_dim: Width of the VLM (prefix) transformer; MEM tokens live in
            that space because they are both an input embedding and a readout
            of the prefix hidden state.
        init_std: Std of the initial-memory parameter. mu-VLA used 0.02, which
            matched the input-embedding scale of its Llama backbone. PaliGemma's
            input embeddings are ~4 (images) to ~10 (language), so the default here
            is larger; see `scripts/probe_embed_scale.py`.
    """

    def __init__(self, num_mem_tokens: int, hidden_dim: int, init_std: float = 0.02) -> None:
        super().__init__()
        self.num_mem_tokens = num_mem_tokens
        self.hidden_dim = hidden_dim
        self.init_std = init_std
        self.initial_memory = nn.Parameter(torch.randn(num_mem_tokens, hidden_dim) * init_std)

    def get_initial_state(self, batch_size: int) -> torch.Tensor:
        """Returns initial memory state for the entire batch: (B, num_mem_tokens, hidden_dim)."""
        return self.initial_memory.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def reset_episodes(
        self,
        mem_state: torch.Tensor,
        is_first: torch.Tensor,
        should_detach: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Per-element episode reset with TBPTT.

        Args:
            mem_state: (B, num_mem_tokens, hidden_dim) - current memory state
            is_first: (B,) bool - True where a new episode begins
            should_detach: (B,) bool or None - True where TBPTT truncation applies.
                If None, defaults to ~is_first (TBPTT length 1: detach every step).

        Returns:
            Updated mem_state where each element [i] is:
              - is_first[i]=True  -> initial_memory (in compute graph)
              - should_detach[i]=True (and not first) -> detached mem_state
              - otherwise -> mem_state as-is (in compute graph, within TBPTT window)
        """
        if should_detach is None:
            should_detach = ~is_first

        batch_size = mem_state.shape[0]
        initial = self.initial_memory.unsqueeze(0).expand(batch_size, -1, -1)
        initial = initial.to(dtype=mem_state.dtype, device=mem_state.device)

        # Use torch.where for clean gradient routing:
        # Priority: is_first > should_detach > keep in graph
        is_first_3d = is_first[:, None, None]
        should_detach_3d = (should_detach & ~is_first)[:, None, None]

        return torch.where(
            is_first_3d,
            initial,
            torch.where(
                should_detach_3d,
                mem_state.detach(),
                mem_state,
            ),
        )

    @staticmethod
    def ema_update(
        mem_input: torch.Tensor,
        mem_output: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        EMA blend producing next step's memory input from current step's input and output.

            M^in_{t+1} = alpha * M^out_t + (1 - alpha) * M^in_t

        Both operands are detached: the EMA ablation intentionally breaks cross-step
        gradient flow through memory. Gradients reach `initial_memory` only via
        within-step attention on `is_first=True` steps.

        Args:
            mem_input:  (B, num_mem_tokens, hidden_dim) - memory fed to the model
                        at step t (output of `reset_episodes`).
            mem_output: (B, num_mem_tokens, hidden_dim) - hidden states read from
                        MEM positions at step t.
            alpha:      Weight on the fresh output. alpha=1 recovers TBPTT-length-1
                        behavior (overwrite with output); alpha=0 freezes memory.

        Returns:
            Detached tensor of shape (B, num_mem_tokens, hidden_dim) to feed into
            the next step (before per-element `reset_episodes`).
        """
        return alpha * mem_output.detach() + (1.0 - alpha) * mem_input.detach()
