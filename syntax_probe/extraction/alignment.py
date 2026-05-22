"""Subword-to-word alignment using HuggingFace offset mappings.

Words are pre-tokenized lists like `["What", "did", "she", "eat", "?"]`. The
sentence is reconstructed by joining words with single spaces. The LLM tokenizer
is then invoked with `return_offsets_mapping=True`, which returns for each
subword its character span `(start, end)` in the joined sentence.

For each word, we compute the *list* of subword indices that overlap the
word's character range ``[word_start, word_start + len(word))``. Downstream
pooling (mean over all subwords, or first subword only) operates on this list.

This handles two distinct offset conventions used by HuggingFace fast tokenizers:

  * SentencePiece-style (e.g., Llama): leading whitespace is stripped before
    offset assignment. The subword for "quick" in " quick" has offsets like
    (4, 9), starting exactly at the first non-space character.

  * Byte-level BPE (e.g., GPT-2, Qwen): leading whitespace is encoded as part
    of the subword token (the special "Ġ" prefix). The subword for " quick"
    in " quick" has offsets like (3, 9), starting at the space character.

A subword belongs to a word iff its character span overlaps the word's
range. For BPE-style tokenizers, the leading-space subword's span (e.g.,
(5, 11) for " world" in "hello world") only overlaps its own word's range
([6, 11)), not the previous word's ([0, 5)) — overlap is strict on the
right-open boundary, so adjacency at the space character does not produce
double-assignment.

Words that end up with zero matched subwords are unalignable; the entire
sentence is then rejected (better to drop than to silently misalign).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Type alias for an offset mapping row: (char_start, char_end), inclusive of start, exclusive of end.
Offset = tuple[int, int]


@dataclass(frozen=True, slots=True)
class WordAlignment:
    """Alignment from word indices to subword positions in an LLM tokenization."""

    sentence: str
    """The reconstructed sentence (words joined by single spaces)."""

    words: tuple[str, ...]
    """The original word-level tokens."""

    word_to_subword_indices: tuple[tuple[int, ...], ...]
    """For each word, the tuple of subword indices that overlap its character range,
    in left-to-right order. Always non-empty for successfully-aligned words."""

    def num_words(self) -> int:
        return len(self.words)

    def first_subword_indices(self) -> tuple[int, ...]:
        """Convenience: for each word, the index of its first subword.

        Equivalent to the legacy "first subword" alignment.
        """
        return tuple(group[0] for group in self.word_to_subword_indices)


def align_words_to_offsets(
    *,
    words: list[str],
    offsets: list[Offset],
    sentence: str,
) -> WordAlignment:
    """Align pre-tokenized words to subword offsets.

    Args:
        words: pre-tokenized words (e.g., from UD or `tokenize_words`).
        offsets: per-subword `(char_start, char_end)` tuples returned by the
            LLM tokenizer's offset mapping. Special tokens with zero-length
            spans (e.g., (0, 0)) are ignored.
        sentence: the joined sentence string the offsets refer to. Must satisfy
            `sentence == " ".join(words)`.

    Raises:
        ValueError: if alignment cannot be performed for some word.
    """
    expected_sentence = " ".join(words)
    if sentence != expected_sentence:
        raise ValueError(
            f"Sentence does not match space-joined words.\n"
            f"  sentence: {sentence!r}\n"
            f"  expected: {expected_sentence!r}"
        )

    # Compute the [start, end) character range of each word in the joined sentence.
    word_ranges: list[tuple[int, int]] = []
    cursor = 0
    for index, word in enumerate(words):
        start = cursor
        end = cursor + len(word)
        word_ranges.append((start, end))
        cursor = end
        if index < len(words) - 1:
            cursor += 1  # the space separator

    # For each word, collect every subword whose span overlaps the word's range.
    # Overlap rule: span [s_start, s_end) and word [w_start, w_end) overlap iff
    # s_start < w_end and s_end > w_start. Special tokens (zero-length spans)
    # are skipped.
    word_to_subword_indices: list[tuple[int, ...]] = []
    for word, (w_start, w_end) in zip(words, word_ranges, strict=True):
        matched: list[int] = []
        for subword_index, (s_start, s_end) in enumerate(offsets):
            if s_end <= s_start:
                continue  # special token, ignore
            if s_start < w_end and s_end > w_start:
                matched.append(subword_index)
        if not matched:
            raise ValueError(
                f"Could not align word {word!r} (range [{w_start}, {w_end})) "
                f"in sentence {sentence!r}. Subword spans: {offsets[:20]}..."
            )
        word_to_subword_indices.append(tuple(matched))

    return WordAlignment(
        sentence=sentence,
        words=tuple(words),
        word_to_subword_indices=tuple(word_to_subword_indices),
    )


__all__ = ["Offset", "WordAlignment", "align_words_to_offsets"]
