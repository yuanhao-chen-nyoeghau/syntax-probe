"""Parsed corpora: schemas, UD EWT loader, gold tree-distance computation."""

from __future__ import annotations

from .distance import compute_dependency_distance_matrix
from .schema import DependencyArc, ParsedCorpus, ParsedSentence
from .ud_ewt import UD_EWT_VERSION, download_ud_ewt, load_ud_ewt_split

__all__ = [
    "DependencyArc",
    "ParsedCorpus",
    "ParsedSentence",
    "UD_EWT_VERSION",
    "compute_dependency_distance_matrix",
    "download_ud_ewt",
    "load_ud_ewt_split",
]
