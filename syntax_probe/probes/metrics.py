"""Probe evaluation metrics: Spearman correlation and UUAS.

These are the two metrics Hewitt & Manning report. Both are computed per-sentence
(in the case of UUAS, per minimum spanning tree) and then aggregated across
sentences.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr


def spearman_correlation(
    predicted_matrices: list[NDArray[np.float32]],
    gold_matrices: list[NDArray[np.int32]],
    *,
    min_length: int = 5,
    max_length: int = 50,
) -> float:
    """Mean of per-sentence Spearman correlations between predicted and gold distances.

    Following H&M, sentences with length outside `[min_length, max_length]` are
    excluded. For each sentence, Spearman is computed over upper-triangular
    pairs only (we avoid double-counting and the diagonal).
    """
    correlations: list[float] = []
    for predicted, gold in zip(predicted_matrices, gold_matrices, strict=True):
        n = predicted.shape[0]
        if not (min_length <= n <= max_length):
            continue
        triu = np.triu_indices(n, k=1)
        if len(triu[0]) == 0:
            continue
        rho, _ = spearmanr(predicted[triu], gold[triu])
        if np.isfinite(rho):
            correlations.append(float(rho))
    if not correlations:
        return float("nan")
    return float(np.mean(correlations))


def undirected_unlabeled_attachment_score(
    predicted_matrices: list[NDArray[np.float32]],
    gold_matrices: list[NDArray[np.int32]],
    *,
    min_length: int = 5,
    max_length: int = 50,
) -> float:
    """UUAS: fraction of gold tree edges recovered by the predicted MST.

    For each sentence, we (a) compute the minimum spanning tree of the predicted
    distance matrix (treated as a complete graph), and (b) compute the gold tree
    edges from gold distance == 1. UUAS is |gold ∩ predicted| / |gold|.

    Both edge sets are undirected (we use sorted pairs) and unlabeled.
    """
    correct = 0
    total = 0
    for predicted, gold in zip(predicted_matrices, gold_matrices, strict=True):
        n = predicted.shape[0]
        if not (min_length <= n <= max_length):
            continue

        gold_edges: set[tuple[int, int]] = set()
        for i in range(n):
            for j in range(i + 1, n):
                if int(gold[i, j]) == 1:
                    gold_edges.add((i, j))

        if not gold_edges:
            continue

        # MST over the upper triangle of `predicted`. Since SciPy's MST takes a
        # CSR-style sparse matrix and treats zero entries as "no edge", we keep
        # the dense matrix but rely on the fact that pairwise distances are >= 0.
        # We add a tiny epsilon to avoid zero-weight edges being dropped.
        weights = predicted.copy()
        np.fill_diagonal(weights, 0.0)
        weights = weights + 1e-9
        mst = minimum_spanning_tree(weights).toarray()
        predicted_edges: set[tuple[int, int]] = set()
        for i in range(n):
            for j in range(n):
                if mst[i, j] > 0:
                    predicted_edges.add((min(i, j), max(i, j)))

        correct += len(gold_edges & predicted_edges)
        total += len(gold_edges)

    if total == 0:
        return float("nan")
    return correct / total


__all__ = [
    "spearman_correlation",
    "undirected_unlabeled_attachment_score",
]
