"""Compute pairwise dependency-tree distance matrices.

This is the gold-target computation for Hewitt & Manning structural probes.
The "distance" between two tokens i and j is the number of edges on the
shortest path between them in the (undirected) dependency tree. This treats
the dependency parse as an undirected graph and computes BFS distances.

Hewitt & Manning's reference implementation does exactly this: see
`structural-probes/task.py` (`ParseDistanceTask.distance_matrix`).
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from .schema import ParsedSentence


def compute_dependency_distance_matrix(parsed: ParsedSentence) -> NDArray[np.int32]:
    """Pairwise undirected dependency-tree distance matrix.

    Returns an ``(n, n)`` int32 matrix where entry ``[i, j]`` is the number of
    edges between tokens ``i`` and ``j`` in the dependency tree. Disconnected
    pairs (if any) are filled with ``n`` as a sentinel; in well-formed UD parses
    the tree is connected and this won't happen.
    """
    n = len(parsed.tokens)
    if n == 0:
        return np.zeros((0, 0), dtype=np.int32)

    # Build an undirected adjacency list. Skip arcs to the root (head_index = -1).
    adj: list[list[int]] = [[] for _ in range(n)]
    for arc in parsed.dependency_arcs:
        if arc.head_index < 0:
            continue
        if not (0 <= arc.head_index < n and 0 <= arc.dependent_index < n):
            continue
        adj[arc.head_index].append(arc.dependent_index)
        adj[arc.dependent_index].append(arc.head_index)

    # BFS from each node.
    matrix = np.full((n, n), fill_value=n, dtype=np.int32)
    for src in range(n):
        matrix[src, src] = 0
        queue: deque[int] = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if matrix[src, v] == n:
                    matrix[src, v] = matrix[src, u] + 1
                    queue.append(v)

    return matrix


__all__ = ["compute_dependency_distance_matrix"]
