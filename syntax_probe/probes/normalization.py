"""Input normalization for the structural probe.

We support four normalization options, selected via
``ProbeConfig.input_normalization``. They all map an activation tensor of shape
``(..., hidden_size)`` to a tensor of the same shape.

Decoder-only LLMs (Llama, Qwen, ...) develop "outlier features" — a small
number of hidden-state dimensions with anomalously large magnitudes — that
cause the structural probe to diverge during training if used raw. The
normalization options below are designed to mitigate this, while preserving
as much structural signal as possible.

* ``"none"``: identity. Use only for ablation.
* ``"l2_norm"``: per-token L2-normalize to unit length. Cheapest fix, fully
  suppresses magnitude blowup, destroys magnitude information.
* ``"per_corpus_standardize"`` (default): subtract per-dimension mean, divide
  by per-dimension std (computed once over the training corpus). Best balance
  for modern decoder-only LLMs.
* ``"per_token_layernorm"``: standardize each token to zero-mean unit-variance
  over the hidden dimension. Equivalent to ``nn.LayerNorm`` with no affine
  parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

if TYPE_CHECKING:
    from ..core.config import ProbeInputNormalization


# Numerical epsilon used by ``per_token_layernorm`` and standardization to avoid
# division by zero on dimensions with zero variance.
NORMALIZATION_EPS = 1e-5


class ProbeInputNormalizer(nn.Module):
    """Pluggable normalizer applied to activations before the probe projection.

    The normalizer is registered as a child of the structural probe so that
    saving and loading the probe carries all required state (in particular,
    the per-corpus mean/std tensors). Identity / l2 / per_token modes have
    zero learnable or buffered state and are essentially free; standardization
    holds two ``(hidden_size,)`` buffers.

    Args:
        kind: which normalization to apply.
        hidden_size: dimensionality of the activations. Required so that
            standardization buffers are sized correctly even before stats
            are loaded.
        mean: optional per-dimension mean (only meaningful for
            ``per_corpus_standardize``). If ``None`` and ``kind`` requires
            stats, the buffers initialize to zero/one as placeholders; the
            normalizer will raise on forward unless stats are set later via
            :meth:`set_corpus_stats`.
        std: optional per-dimension std (same conditions as ``mean``).
    """

    def __init__(
        self,
        *,
        kind: ProbeInputNormalization,
        hidden_size: int,
        mean: NDArray[np.float32] | None = None,
        std: NDArray[np.float32] | None = None,
    ) -> None:
        super().__init__()
        if kind not in ("none", "l2_norm", "per_corpus_standardize", "per_token_layernorm"):
            raise ValueError(f"Unknown probe input normalization kind: {kind!r}")
        self.kind: ProbeInputNormalization = kind
        self.hidden_size = hidden_size
        # Stats are always allocated so that ``state_dict`` shape is stable;
        # they are only consulted in the per_corpus_standardize branch.
        self.register_buffer(
            "corpus_mean",
            torch.zeros(hidden_size, dtype=torch.float32),
        )
        self.register_buffer(
            "corpus_std",
            torch.ones(hidden_size, dtype=torch.float32),
        )
        # Whether per-corpus stats have been set. Saved as a buffer so it
        # survives serialization. Stored as a 0-D long tensor so checkpointing
        # is uniform (no Python bool gymnastics).
        self.register_buffer(
            "corpus_stats_initialized",
            torch.zeros((), dtype=torch.long),
        )
        if mean is not None and std is not None:
            self.set_corpus_stats(mean=mean, std=std)
        elif mean is not None or std is not None:
            raise ValueError("Provide both `mean` and `std`, or neither.")

    def set_corpus_stats(
        self,
        *,
        mean: NDArray[np.float32],
        std: NDArray[np.float32],
    ) -> None:
        """Install per-dimension corpus mean and std (used by per_corpus_standardize)."""
        if mean.shape != (self.hidden_size,):
            raise ValueError(
                f"Expected mean shape ({self.hidden_size},), got {mean.shape}"
            )
        if std.shape != (self.hidden_size,):
            raise ValueError(
                f"Expected std shape ({self.hidden_size},), got {std.shape}"
            )
        # Clamp std to avoid division by zero on dead dimensions.
        std_clamped = np.maximum(std.astype(np.float32), NORMALIZATION_EPS)
        self.corpus_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.corpus_std.copy_(torch.from_numpy(std_clamped))
        self.corpus_stats_initialized.fill_(1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize ``hidden_states`` of shape ``(..., hidden_size)``."""
        if self.kind == "none":
            return hidden_states

        if self.kind == "l2_norm":
            # Per-token L2 normalization. ``norm`` keeps the leading shape and
            # broadcasts back onto the hidden dim.
            norm = hidden_states.norm(dim=-1, keepdim=True).clamp(min=NORMALIZATION_EPS)
            return hidden_states / norm

        if self.kind == "per_token_layernorm":
            # Mean and variance over the hidden dimension, per token.
            mean = hidden_states.mean(dim=-1, keepdim=True)
            var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
            return (hidden_states - mean) / torch.sqrt(var + NORMALIZATION_EPS)

        if self.kind == "per_corpus_standardize":
            if int(self.corpus_stats_initialized.item()) == 0:
                raise RuntimeError(
                    "per_corpus_standardize requires corpus statistics to be set "
                    "via set_corpus_stats() before forward()."
                )
            mean = self.corpus_mean.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            std = self.corpus_std.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            return (hidden_states - mean) / std

        # Should be unreachable due to __init__ validation.
        raise ValueError(f"Unknown normalization kind: {self.kind!r}")


def compute_corpus_stats(
    activations: NDArray[np.float32],
    *,
    sample_limit: int | None = 200_000,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute per-dimension mean and std over a flat activation array.

    Args:
        activations: shape ``(num_word_vectors, hidden_size)`` float32 array.
            All vectors contribute to the statistics; rows are not weighted.
        sample_limit: if provided and the number of rows exceeds this, take a
            uniform random sample of that size before computing statistics.
            ``None`` to use all rows. Default 200K is enough for stable stats
            on a ~25K-sentence corpus and avoids large-allocation pitfalls.

    Returns:
        ``(mean, std)`` each of shape ``(hidden_size,)`` float32.
    """
    if activations.ndim != 2:
        raise ValueError(
            f"Expected 2D activations (n_words, hidden_size), got shape {activations.shape}"
        )
    n = activations.shape[0]
    if sample_limit is not None and n > sample_limit:
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(n, size=sample_limit, replace=False)
        sample = activations[indices]
    else:
        sample = activations
    mean = sample.mean(axis=0).astype(np.float32)
    std = sample.std(axis=0, ddof=0).astype(np.float32)
    return mean, std


__all__ = [
    "NORMALIZATION_EPS",
    "ProbeInputNormalizer",
    "compute_corpus_stats",
]
