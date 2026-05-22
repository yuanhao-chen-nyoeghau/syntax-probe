"""Decoder-only LLM activation extractor.

For each input sentence (provided as a list of pre-tokenized words), runs the
LLM forward pass with `output_hidden_states=True`, aligns subwords back to
words using offset mappings, and returns per-word per-layer activations.

Activations are returned as a `(num_layers, num_words, hidden_size)` array per
sentence. The number of layers includes the embedding layer (layer 0) plus
each transformer block, which is the standard `output_hidden_states` shape.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ..core.config import ModelConfig
from .alignment import align_words_to_offsets

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LayerActivations:
    """One sentence's activations across all layers."""

    sentence_id: str
    words: list[str]
    activations: NDArray[np.float32]
    """Shape: (num_layers, num_words, hidden_size)."""

    def num_layers(self) -> int:
        return int(self.activations.shape[0])

    def num_words(self) -> int:
        return int(self.activations.shape[1])

    def hidden_size(self) -> int:
        return int(self.activations.shape[2])


@dataclass(slots=True)
class ExtractionResult:
    """Output of an extraction batch."""

    items: list[LayerActivations]
    skipped_sentence_ids: list[str]
    """Sentences whose subword-to-word alignment failed; not included in `items`."""


class LLMActivationExtractor:
    """Extract per-word per-layer hidden states from a HuggingFace causal LM."""

    def __init__(self, model: Any, tokenizer: Any, config: ModelConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def from_pretrained(cls, config: ModelConfig) -> LLMActivationExtractor:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name or config.model_name,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            use_fast=True,
        )
        if not tokenizer.is_fast:
            raise RuntimeError(
                f"Tokenizer for {config.model_name!r} is not a fast tokenizer; "
                f"offset mapping is required for subword alignment."
            )

        # Llama-family tokenizers ship without a pad_token because pretraining
        # didn't need padding. The extractor calls tokenizer(..., padding=True),
        # which transformers refuses without a pad_token. Setting it to eos_token
        # is the standard fix and is safe here because:
        #   (a) we only do forward passes (no generation, no training), and
        #   (b) the attention_mask returned alongside input_ids correctly
        #       excludes padding positions from the model's computation,
        #       regardless of which token id is used as the pad.
        # Qwen tokenizers already define pad_token, so this branch is a no-op
        # for them.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(
                "Tokenizer had no pad_token; set pad_token = eos_token (%r)",
                tokenizer.eos_token,
            )

        torch_dtype = _resolve_dtype(config.torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=config.device_map,
        )
        model.eval()
        return cls(model=model, tokenizer=tokenizer, config=config)

    @torch.inference_mode()
    def extract(
        self,
        items: list[tuple[str, list[str]]],
        *,
        progress: bool = True,
    ) -> ExtractionResult:
        """Extract activations for a list of `(sentence_id, words)` pairs.

        The sentence is reconstructed as `" ".join(words)` and tokenized with
        the LLM tokenizer. Words that fail to align (rare; usually special-token
        edge cases) cause the whole sentence to be skipped with a logged warning.
        """
        results: list[LayerActivations] = []
        skipped: list[str] = []

        # Length-bucketing: sort items by word-count (descending) before
        # batching. The tokenizer call uses padding="longest" (`padding=True`
        # below), so each batch only pads to its own longest sentence.
        # Grouping similar-length sentences therefore minimizes padding
        # waste -- on UD-EWT, average sentence is ~13 words but the long
        # tail extends past 100, so an unsorted batch can spend the bulk
        # of its compute on [PAD] positions.
        #
        # Descending order surfaces OOM at the first (largest) batch instead
        # of after hours of work. Ordering does not affect correctness:
        # downstream code re-keys by sentence_id via cache metadata, and
        # each sentence's hidden states are independent of others in the
        # batch (attention is masked over padding).
        items = sorted(items, key=lambda it: len(it[1]), reverse=True)

        batches = list(_batched(items, self.config.batch_size))
        iterator: Iterator[list[tuple[str, list[str]]]] = (
            tqdm(batches, desc="extracting", unit="batch") if progress else iter(batches)
        )

        for batch in iterator:
            batch_results, batch_skipped = self._extract_batch(batch)
            results.extend(batch_results)
            skipped.extend(batch_skipped)

        if skipped:
            logger.warning(
                "Skipped %d / %d sentences due to alignment failures.",
                len(skipped),
                len(items),
            )

        return ExtractionResult(items=results, skipped_sentence_ids=skipped)

    def _extract_batch(
        self, batch: list[tuple[str, list[str]]]
    ) -> tuple[list[LayerActivations], list[str]]:
        sentences = [" ".join(words) for _, words in batch]

        encoded = self.tokenizer(
            sentences,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
        )
        offset_mapping = encoded.pop("offset_mapping")

        device = _model_device(self.model)
        model_inputs = {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}
        outputs = self.model(**model_inputs, output_hidden_states=True, use_cache=False)

        # outputs.hidden_states is a tuple of length L+1 (embedding output + each block).
        # Each tensor is (batch, seq_len, hidden_size).
        # We stack along a new "layer" axis to get (L+1, batch, seq_len, hidden_size).
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
        hidden_states_cpu = hidden_states.detach().to(torch.float32).cpu().numpy()

        results: list[LayerActivations] = []
        skipped: list[str] = []

        for batch_index, (sentence_id, words) in enumerate(batch):
            sentence = sentences[batch_index]
            offsets_row = offset_mapping[batch_index].tolist()  # list of [start, end]
            offsets = [(int(s), int(e)) for s, e in offsets_row]

            try:
                alignment = align_words_to_offsets(words=words, offsets=offsets, sentence=sentence)
            except ValueError as err:
                logger.debug("Alignment failed for %s: %s", sentence_id, err)
                skipped.append(sentence_id)
                continue

            # Pool subwords into per-word vectors. Pooling mode is configured
            # on ModelConfig.subword_pooling; "mean" matches H&M (2019)
            # footnote 4 and is the recommended default.
            per_word = _pool_subwords(
                hidden_states=hidden_states_cpu,
                batch_index=batch_index,
                word_to_subword_indices=alignment.word_to_subword_indices,
                pooling=self.config.subword_pooling,
            )
            # per_word shape: (num_layers, num_words, hidden_size)

            results.append(
                LayerActivations(
                    sentence_id=sentence_id,
                    words=list(words),
                    activations=per_word,
                )
            )

        return results, skipped


def _pool_subwords(
    *,
    hidden_states: NDArray[np.float32],
    batch_index: int,
    word_to_subword_indices: tuple[tuple[int, ...], ...],
    pooling: str,
) -> NDArray[np.float32]:
    """Pool subword hidden states into per-word vectors for one sentence.

    Args:
        hidden_states: full batch hidden states, shape (num_layers, batch, seq_len, hidden_size).
        batch_index: which sentence in the batch.
        word_to_subword_indices: per-word tuples of subword indices to combine.
        pooling: "mean" or "first".

    Returns:
        Array of shape (num_layers, num_words, hidden_size).
    """
    if pooling == "first":
        first_indices = np.asarray([group[0] for group in word_to_subword_indices], dtype=np.int64)
        return hidden_states[:, batch_index, first_indices, :]

    if pooling == "mean":
        num_layers = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]
        num_words = len(word_to_subword_indices)
        out = np.empty((num_layers, num_words, hidden_size), dtype=np.float32)
        sentence_states = hidden_states[:, batch_index, :, :]  # (num_layers, seq_len, hidden_size)
        for word_index, group in enumerate(word_to_subword_indices):
            sub = sentence_states[:, list(group), :]  # (num_layers, k, hidden_size)
            out[:, word_index, :] = sub.mean(axis=1)
        return out

    raise ValueError(f"Unknown subword pooling mode: {pooling!r}")


def _batched(
    items: list[tuple[str, list[str]]], batch_size: int
) -> Iterator[list[tuple[str, list[str]]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _resolve_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _model_device(model: Any) -> torch.device:
    """Best-effort device discovery; works for single-device and device_map='auto'."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


__all__ = ["ExtractionResult", "LayerActivations", "LLMActivationExtractor"]
