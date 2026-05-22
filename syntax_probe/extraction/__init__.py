"""LLM activation extraction with subword-to-word alignment.

For each input sentence, we extract per-word hidden states at every layer of
the LLM. Words are pre-tokenized (either by UD or by `tokenize_words`); the LLM
tokenizer's offset mapping is used to align subwords to words, with the
*first-subword* convention (matching Hewitt & Manning).

Activations are cached to disk in a sharded format so that probes can be trained
without re-running the LLM forward pass.
"""

from __future__ import annotations

from .alignment import WordAlignment, align_words_to_offsets
from .cache import ActivationCache
from .extractor import ExtractionResult, LayerActivations, LLMActivationExtractor

__all__ = [
    "ActivationCache",
    "ExtractionResult",
    "LayerActivations",
    "LLMActivationExtractor",
    "WordAlignment",
    "align_words_to_offsets",
]
