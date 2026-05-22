"""Hewitt & Manning structural probes.

A structural probe learns a linear map ``B`` from LLM hidden states to a
low-rank space such that squared Euclidean distances in that space approximate
gold dependency-tree distances:

    d_B(h_i, h_j)^2 = || B(h_i - h_j) ||^2  ≈  d_T(w_i, w_j)

The training loss is L1 between predicted and gold distances, normalized per
sentence by the squared sentence length (so all sentences contribute equally
regardless of length).

References:
    Hewitt, J. & Manning, C. D. (2019). A Structural Probe for Finding Syntax in
    Word Representations. NAACL.
    https://github.com/john-hewitt/structural-probes
"""

from __future__ import annotations

from .data import ProbeBatch, ProbeTrainingData, build_probe_training_data
from .metrics import (
    spearman_correlation,
    undirected_unlabeled_attachment_score,
)
from .normalization import ProbeInputNormalizer, compute_corpus_stats
from .structural import StructuralProbe
from .training import LayerProbeBank, ProbeTrainResult, train_layer_probe_bank, train_probe

__all__ = [
    "LayerProbeBank",
    "ProbeBatch",
    "ProbeInputNormalizer",
    "ProbeTrainingData",
    "ProbeTrainResult",
    "StructuralProbe",
    "build_probe_training_data",
    "compute_corpus_stats",
    "spearman_correlation",
    "train_layer_probe_bank",
    "train_probe",
    "undirected_unlabeled_attachment_score",
]
