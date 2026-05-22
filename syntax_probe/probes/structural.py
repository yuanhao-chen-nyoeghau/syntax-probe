"""The Hewitt & Manning structural probe.

The probe is a single linear projection ``B`` of shape ``(rank, hidden_size)``,
applied to *differences* between word vectors. The squared L2 norm of the
projected difference is the predicted dependency-tree distance:

    d_B(h_i, h_j)^2 = ||B(f(h_i) - f(h_j))||^2

where ``f`` is an input normalizer (see :mod:`probes.normalization`).

Training:
    For each sentence of length n, the L1 loss between predicted and gold
    pairwise distances is averaged over the n^2 pairs. Then sentences are
    averaged uniformly. This matches the H&M reference implementation
    (see `structural-probes/loss.py`, `L1DistanceLoss`).
"""

from __future__ import annotations

import torch
from torch import nn

from .normalization import ProbeInputNormalizer


class StructuralProbe(nn.Module):
    """Linear probe predicting pairwise dependency-tree distances.

    Args:
        hidden_size: dimensionality of the LLM hidden states.
        rank: dimensionality of the projection target. The probe is over-parameterized
            if rank == hidden_size; H&M typically use rank=32 for ELMo (1024-d) and
            rank=128 for BERT-large. We default to 64 for modern LLMs (~2048-4096 d).
        normalizer: optional input normalizer applied before the linear projection.
            If ``None``, a no-op normalizer is constructed (kind="none"). Pass a
            normalizer with corpus stats already set when using
            ``per_corpus_standardize`` so the probe is self-contained at save/load.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        normalizer: ProbeInputNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        if normalizer is None:
            normalizer = ProbeInputNormalizer(kind="none", hidden_size=hidden_size)
        if normalizer.hidden_size != hidden_size:
            raise ValueError(
                f"Normalizer hidden_size={normalizer.hidden_size} does not match "
                f"probe hidden_size={hidden_size}."
            )
        self.normalizer = normalizer
        # Bias-free linear layer matching H&M's parameterization.
        self.projection = nn.Linear(hidden_size, rank, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute pairwise predicted squared distances.

        Args:
            hidden_states: (batch, seq_len, hidden_size).

        Returns:
            Predicted squared distances, shape (batch, seq_len, seq_len). The
            diagonal is zero by construction.
        """
        # Apply input normalization first (suppresses outlier-feature blowup
        # in modern decoder-only LLMs).
        normalized = self.normalizer(hidden_states)
        # Project: (batch, seq_len, rank).
        projected = self.projection(normalized)
        # Compute pairwise differences via broadcasting:
        #   diff[b, i, j] = projected[b, i] - projected[b, j]
        # Shape: (batch, seq_len, seq_len, rank).
        diff = projected.unsqueeze(2) - projected.unsqueeze(1)
        # Squared L2 norm along the last axis.
        return diff.pow(2).sum(dim=-1)


def l1_distance_loss(
    predicted: torch.Tensor,
    gold: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """L1 loss between predicted and gold pairwise distances, with H&M normalization.

    Args:
        predicted: (batch, max_len, max_len) predicted squared distances.
        gold: (batch, max_len, max_len) gold tree distances.
        lengths: (batch,) sentence lengths.

    Returns:
        Scalar loss = mean over sentences of (per-sentence L1 sum / n^2).
    """
    batch_size, max_len, _ = predicted.shape
    device = predicted.device

    # Build (batch, max_len, max_len) mask: pair (i, j) is valid iff i, j < length.
    length_range = torch.arange(max_len, device=device)
    valid_token = length_range.unsqueeze(0) < lengths.unsqueeze(1)  # (batch, max_len)
    valid_pair = valid_token.unsqueeze(2) & valid_token.unsqueeze(1)  # (batch, max_len, max_len)

    abs_error = (predicted - gold).abs() * valid_pair.float()
    per_sentence_sum = abs_error.view(batch_size, -1).sum(dim=-1)  # (batch,)
    # Divide by n^2 to match H&M (each sentence contributes equally regardless of length).
    n_squared = lengths.float().pow(2).clamp(min=1.0)
    per_sentence_loss = per_sentence_sum / n_squared
    return per_sentence_loss.mean()


__all__ = ["StructuralProbe", "l1_distance_loss"]
