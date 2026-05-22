"""Text utilities: word tokenization and subsequence search."""

from __future__ import annotations

import re

# Word characters or any single non-word non-whitespace character (i.e., punctuation
# tokens, each one its own token). This matches what UD-style tokenization produces
# for English text in most cases.
_WORD_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize_words(text: str) -> list[str]:
    """Split text into word and punctuation tokens.

    Examples:
        >>> tokenize_words("What did she eat?")
        ['What', 'did', 'she', 'eat', '?']
        >>> tokenize_words("the teacher")
        ['the', 'teacher']
    """
    return _WORD_PATTERN.findall(text)


def find_subsequence(haystack: list[str], needle: list[str]) -> int | None:
    """Return the start index of `needle` in `haystack`, or `None` if not found.

    Searches for the first occurrence of the contiguous subsequence ``needle``
    inside ``haystack`` using exact equality on tokens.
    """
    if not needle:
        return None
    nlen = len(needle)
    for i in range(len(haystack) - nlen + 1):
        if haystack[i : i + nlen] == needle:
            return i
    return None


__all__ = ["find_subsequence", "tokenize_words"]
