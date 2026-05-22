"""Load and download the UD English EWT treebank.

UD EWT ships pre-parsed in CoNLL-U format. We don't need a parser at all — we
just convert the CoNLL-U files to our `ParsedSentence` schema. This is the
fastest and most reliable path to gold dependency trees, and it's what
Hewitt & Manning's reference repo uses in its example pipeline.

References:
- UD English EWT: https://github.com/UniversalDependencies/UD_English-EWT
- Hewitt & Manning structural-probes: https://github.com/john-hewitt/structural-probes
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import conllu

from .schema import DependencyArc, ParsedCorpus, ParsedSentence, Split

logger = logging.getLogger(__name__)

# UD English EWT version. The repo only has a single GitHub release tag (r1.0,
# from 2019) but master is continuously updated. As of writing, master tracks
# UD v2.17 (2025-11-15). We pull from master so this always corresponds to the
# latest published EWT data; if we want a specific commit pin, replace 'master'
# below with a commit hash.
UD_EWT_VERSION = "master"
UD_EWT_BASE_URL = (
    f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{UD_EWT_VERSION}"
)

_SPLIT_FILES: dict[Split, str] = {
    "train": "en_ewt-ud-train.conllu",
    "dev": "en_ewt-ud-dev.conllu",
    "test": "en_ewt-ud-test.conllu",
}


def download_ud_ewt(output_dir: Path) -> dict[Split, Path]:
    """Download all three EWT splits into ``output_dir``.

    Idempotent: if a file already exists at the expected path, it is not re-downloaded.
    Returns a mapping from split name to local file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[Split, Path] = {}
    for split, filename in _SPLIT_FILES.items():
        local_path = output_dir / filename
        if not local_path.exists():
            url = f"{UD_EWT_BASE_URL}/{filename}"
            logger.info("Downloading %s -> %s", url, local_path)
            urllib.request.urlretrieve(url, local_path)
        else:
            logger.debug("Already have %s", local_path)
        paths[split] = local_path
    return paths


def load_ud_ewt_split(
    conllu_path: Path,
    *,
    split: Split,
    min_length: int = 3,
    max_length: int | None = 60,
) -> list[ParsedSentence]:
    """Load a single CoNLL-U file into a list of `ParsedSentence`.

    Args:
        conllu_path: path to the .conllu file.
        split: split label to record on each sentence.
        min_length: skip sentences shorter than this many tokens.
        max_length: skip sentences longer than this many tokens; ``None`` for no limit.

    Skipping rules:
        * Multi-word tokens (e.g., "don't" expanded to "do" + "n't") are handled by
          using the syntactic-word level — i.e., we use the expanded form, since
          that's what the parse references. CoNLL-U represents this with composite
          IDs like ``5-6``; we drop those rows and keep only the integer-ID rows
          that have actual analyses attached.
        * Sentences with elided / empty nodes (decimal IDs like ``8.1``) are skipped
          entirely, because they introduce phantom tokens not present in the
          surface string.
    """
    text = conllu_path.read_text(encoding="utf-8")
    sentences: list[ParsedSentence] = []

    for index, token_list in enumerate(conllu.parse(text)):
        sent_id = token_list.metadata.get("sent_id", f"{conllu_path.stem}-{index:06d}")

        # Detect elided/empty nodes (decimal IDs). Skip these sentences entirely:
        # the parse references tokens that don't appear in surface text.
        has_elided = any(isinstance(tok["id"], tuple) and len(tok["id"]) == 3 for tok in token_list)
        if has_elided:
            continue

        # Keep only single-integer-ID rows (drop CoNLL-U multi-word-token rows like "5-6").
        regular_tokens = [tok for tok in token_list if isinstance(tok["id"], int)]

        n = len(regular_tokens)
        if n < min_length:
            continue
        if max_length is not None and n > max_length:
            continue

        # Map UD's 1-based IDs to our 0-based indices.
        id_to_index = {tok["id"]: idx for idx, tok in enumerate(regular_tokens)}

        tokens: list[str] = [tok["form"] for tok in regular_tokens]
        arcs: list[DependencyArc] = []

        valid = True
        for tok in regular_tokens:
            head_id = tok["head"]
            dep_id = tok["id"]
            dep_idx = id_to_index[dep_id]

            if head_id == 0:
                head_idx = -1  # root
            elif head_id in id_to_index:
                head_idx = id_to_index[head_id]
            else:
                # Head points to an unknown id (rare CoNLL-U inconsistency); skip sentence.
                valid = False
                break

            arcs.append(
                DependencyArc(
                    head_index=head_idx,
                    dependent_index=dep_idx,
                    relation=tok.get("deprel", "dep"),
                )
            )

        if not valid:
            continue

        sentences.append(
            ParsedSentence(
                sentence_id=sent_id,
                text=" ".join(tokens),
                tokens=tokens,
                dependency_arcs=arcs,
                split=split,
                metadata={"source_file": conllu_path.name},
            )
        )

    logger.info("Loaded %d sentences from %s (split=%s)", len(sentences), conllu_path.name, split)
    return sentences


def load_full_ud_ewt(
    download_dir: Path,
    *,
    min_length: int = 3,
    max_length: int | None = 60,
) -> ParsedCorpus:
    """Download (if needed) and load all three splits into a single corpus."""
    paths = download_ud_ewt(download_dir)
    all_sentences: list[ParsedSentence] = []
    for split, path in paths.items():
        all_sentences.extend(
            load_ud_ewt_split(path, split=split, min_length=min_length, max_length=max_length)
        )

    return ParsedCorpus(
        name=f"ud_ewt_{UD_EWT_VERSION}",
        sentences=all_sentences,
        source=f"UD_English-EWT@{UD_EWT_VERSION}",
    )


__all__ = [
    "UD_EWT_VERSION",
    "download_ud_ewt",
    "load_full_ud_ewt",
    "load_ud_ewt_split",
]
