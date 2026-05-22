"""Data containers for probe training and evaluation.

A `ProbeTrainingData` is a per-split bundle of:
- Per-sentence activations at one layer (variable-length).
- Per-sentence gold distance matrices.

We pad activations and distance matrices to the same per-batch length so that
training can be vectorized. The padding is masked out in the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from ..corpora.distance import compute_dependency_distance_matrix
from ..corpora.schema import ParsedSentence
from ..extraction.cache import CachedSplit


@dataclass(slots=True)
class ProbeTrainingData:
    """Per-layer probe-training data for one split.

    Attributes:
        sentence_ids: list of sentence ids in order.
        activations: list of (n_words, hidden_size) float32 arrays.
        gold_distances: list of (n_words, n_words) int32 matrices.
        hidden_size: dimensionality of LLM hidden states.
    """

    sentence_ids: list[str]
    activations: list[NDArray[np.float32]]
    gold_distances: list[NDArray[np.int32]]
    hidden_size: int

    def __len__(self) -> int:
        return len(self.sentence_ids)


@dataclass(slots=True)
class ProbeBatch:
    """One batch ready for the probe forward pass."""

    activations: torch.Tensor  # (batch, max_len, hidden_size), float
    gold_distances: torch.Tensor  # (batch, max_len, max_len), float
    lengths: torch.Tensor  # (batch,), int

    def to(self, device: torch.device | str) -> ProbeBatch:
        return ProbeBatch(
            activations=self.activations.to(device),
            gold_distances=self.gold_distances.to(device),
            lengths=self.lengths.to(device),
        )


def build_probe_training_data(
    *,
    parsed_sentences: list[ParsedSentence],
    cached_split: CachedSplit,
) -> ProbeTrainingData:
    """Pair parsed sentences with their cached activations to build training data.

    Lookup is by ``sentence_id`` — the cache and parsed-sentence list need not
    be in the same order. (The extractor uses length-bucketing to minimize
    padding waste, which reorders sentences relative to the input corpus.
    Sentence-id keying keeps that optimization invisible to consumers.)

    The output is in ``parsed_sentences`` order. Sentences whose word counts
    disagree between parse and cache, or whose ids appear on only one side,
    cause an error.
    """
    if len(parsed_sentences) != cached_split.num_sentences():
        raise ValueError(
            f"Mismatched sentence counts: parsed={len(parsed_sentences)}, "
            f"cached={cached_split.num_sentences()}"
        )

    # Map sentence_id -> position in the cache, so we can slice activations
    # regardless of the order they were written in.
    id_to_cache_index: dict[str, int] = {
        sid: i for i, sid in enumerate(cached_split.sentence_ids)
    }

    activations: list[NDArray[np.float32]] = []
    gold_distances: list[NDArray[np.int32]] = []
    sentence_ids: list[str] = []
    hidden_size: int | None = None

    for parsed in parsed_sentences:
        cache_index = id_to_cache_index.get(parsed.sentence_id)
        if cache_index is None:
            raise ValueError(
                f"Sentence {parsed.sentence_id!r} appears in the parsed corpus "
                f"but is not present in the cached split."
            )
        sentence_activations = cached_split.slice_for_sentence(cache_index)
        if len(parsed.tokens) != sentence_activations.shape[0]:
            raise ValueError(
                f"Sentence {parsed.sentence_id}: parsed has {len(parsed.tokens)} tokens "
                f"but cache has {sentence_activations.shape[0]} word vectors."
            )

        activations.append(sentence_activations)
        gold_distances.append(compute_dependency_distance_matrix(parsed))
        sentence_ids.append(parsed.sentence_id)
        if hidden_size is None:
            hidden_size = sentence_activations.shape[1]

    if hidden_size is None:
        hidden_size = 0

    return ProbeTrainingData(
        sentence_ids=sentence_ids,
        activations=activations,
        gold_distances=gold_distances,
        hidden_size=hidden_size,
    )


def collate_probe_batch(
    indices: list[int],
    data: ProbeTrainingData,
) -> ProbeBatch:
    """Build a padded batch from a list of indices into `data`."""
    batch_size = len(indices)
    if batch_size == 0:
        raise ValueError("Cannot collate an empty batch")

    lengths_list = [data.activations[i].shape[0] for i in indices]
    max_len = max(lengths_list)
    hidden_size = data.hidden_size

    activations = np.zeros((batch_size, max_len, hidden_size), dtype=np.float32)
    gold = np.zeros((batch_size, max_len, max_len), dtype=np.float32)
    for batch_index, sent_index in enumerate(indices):
        n = lengths_list[batch_index]
        activations[batch_index, :n, :] = data.activations[sent_index]
        gold[batch_index, :n, :n] = data.gold_distances[sent_index].astype(np.float32)

    return ProbeBatch(
        activations=torch.from_numpy(activations),
        gold_distances=torch.from_numpy(gold),
        lengths=torch.tensor(lengths_list, dtype=torch.int64),
    )


__all__ = ["ProbeBatch", "ProbeTrainingData", "build_probe_training_data", "collate_probe_batch"]
