"""Probe-training experiment runner.

Pipeline:
1. Load parsed corpus (UD EWT splits) from disk.
2. Extract LLM activations for every sentence in train + dev splits at every layer.
   (Cached: skip extraction if a cache for this (corpus, model) already exists.)
3. If using per-corpus standardization, compute and cache per-layer mean/std stats.
4. Build per-layer probe-training data (activations + gold distance matrices).
5. Train one structural probe per layer with H&M LR-reset schedule, saving
   per-epoch loss curves for each layer.
6. Save the trained probe bank (including normalizer state).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ..core.config import AppConfig, ProbeTrainingConfig
from ..core.context import RunContext, write_manifest
from ..core.io import write_json
from ..core.seed import seed_everything
from ..corpora.schema import ParsedCorpus, ParsedSentence
from ..extraction.cache import ActivationCache, make_cache_key
from ..extraction.extractor import LLMActivationExtractor
from ..probes.data import ProbeTrainingData, build_probe_training_data
from ..probes.normalization import ProbeInputNormalizer, compute_corpus_stats
from ..probes.training import train_layer_probe_bank
from .base import ExperimentResult, ExperimentRunner
from .registry import register_experiment_runner

logger = logging.getLogger(__name__)


class ProbeTrainingRunner(ExperimentRunner):
    kind = "probe_training"

    def run(self, *, app_config: AppConfig, context: RunContext) -> ExperimentResult:
        if not isinstance(app_config.experiment, ProbeTrainingConfig):
            raise TypeError("Expected probe_training config")

        seed_everything(app_config.runtime.seed)
        cfg = app_config.experiment

        # 1. Load parsed corpus.
        corpus = _load_parsed_corpus_jsonl(cfg.corpus_path)
        logger.info("Loaded %d parsed sentences from %s", len(corpus.sentences), cfg.corpus_path)

        train_sentences = corpus.by_split("train")
        dev_sentences = corpus.by_split("dev")
        if not train_sentences or not dev_sentences:
            raise ValueError(
                f"Corpus is missing train/dev splits "
                f"(train={len(train_sentences)}, dev={len(dev_sentences)})"
            )

        # 2. Set up the activation cache.
        cache = ActivationCache(
            make_cache_key(
                corpus_name=cfg.activation_cache_name,
                model_name=app_config.model.model_name,
                cache_root=context.cache_dir,
            )
        )
        logger.info("Activation cache: %s", cache.directory)

        # 3. Extract activations if not already cached.
        train_sentences = _ensure_cache(
            cache=cache,
            split="train",
            sentences=train_sentences,
            app_config=app_config,
        )
        dev_sentences = _ensure_cache(
            cache=cache,
            split="dev",
            sentences=dev_sentences,
            app_config=app_config,
        )

        # 4. Determine layers.
        num_layers = cache.num_layers("train")
        all_layer_indices = list(range(num_layers))
        layer_indices = cfg.layers if cfg.layers is not None else all_layer_indices
        for idx in layer_indices:
            if idx not in all_layer_indices:
                raise ValueError(
                    f"Requested layer {idx} but corpus cache has only {num_layers} layers"
                )
        logger.info("Training probes for layers: %s", layer_indices)

        # 5. Build per-layer probe-training data.
        train_data_per_layer = {
            layer_index: _build_layer_data(cache, "train", train_sentences, layer_index)
            for layer_index in layer_indices
        }
        dev_data_per_layer = {
            layer_index: _build_layer_data(cache, "dev", dev_sentences, layer_index)
            for layer_index in layer_indices
        }

        # 6. Build per-layer normalizers. For per_corpus_standardize, this
        #    requires per-layer stats (computed from the train cache); for
        #    other kinds the normalizer is stateless and built directly.
        normalization_kind = cfg.probe.input_normalization
        normalizers_per_layer: dict[int, ProbeInputNormalizer] = {}
        if normalization_kind == "per_corpus_standardize":
            _ensure_corpus_stats(
                cache=cache,
                split="train",
                layer_indices=layer_indices,
                train_data_per_layer=train_data_per_layer,
            )
            for layer_index in layer_indices:
                mean, std = cache.read_corpus_stats(split="train", layer_index=layer_index)
                hidden_size = train_data_per_layer[layer_index].hidden_size
                normalizers_per_layer[layer_index] = ProbeInputNormalizer(
                    kind=normalization_kind,
                    hidden_size=hidden_size,
                    mean=mean,
                    std=std,
                )
        else:
            for layer_index in layer_indices:
                hidden_size = train_data_per_layer[layer_index].hidden_size
                normalizers_per_layer[layer_index] = ProbeInputNormalizer(
                    kind=normalization_kind,
                    hidden_size=hidden_size,
                )

        # 7. Train probes.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        history_dir = context.artifact_dir / "training_history"
        bank = train_layer_probe_bank(
            layer_indices=layer_indices,
            train_data_per_layer=train_data_per_layer,
            dev_data_per_layer=dev_data_per_layer,
            config=cfg.probe,
            seed=app_config.runtime.seed,
            device=device,
            normalizers_per_layer=normalizers_per_layer,
            history_dir=history_dir,
        )

        # 8. Save the trained probe bank.
        bank_dir = context.artifact_dir / "probe_bank"
        bank.save(bank_dir)

        # Save corpus + cache pointer plus per-run config so apply-probes can
        # reproduce the setup, including pooling and normalization choices.
        provenance_path = context.artifact_dir / "training_provenance.json"
        write_json(
            provenance_path,
            {
                "corpus_path": str(cfg.corpus_path),
                "cache_directory": str(cache.directory),
                "model_name": app_config.model.model_name,
                "subword_pooling": app_config.model.subword_pooling,
                "input_normalization": normalization_kind,
                "layer_indices": layer_indices,
                "num_train_sentences": len(train_sentences),
                "num_dev_sentences": len(dev_sentences),
            },
        )

        manifest_path = write_manifest(
            context,
            {
                "num_layers": num_layers,
                "layers_trained": layer_indices,
                "num_train_sentences": len(train_sentences),
                "num_dev_sentences": len(dev_sentences),
                "subword_pooling": app_config.model.subword_pooling,
                "input_normalization": normalization_kind,
                "best_dev_uuas_by_layer": {
                    str(idx): bank.evaluations[idx].dev_uuas for idx in layer_indices
                },
                "artifacts": {
                    "probe_bank": str(bank_dir),
                    "training_provenance": str(provenance_path),
                    "training_history": str(history_dir),
                },
            },
        )

        return ExperimentResult(
            experiment_kind=self.kind,
            run_id=context.run_id,
            summary={
                "layers_trained": layer_indices,
                "best_dev_uuas_by_layer": {
                    idx: bank.evaluations[idx].dev_uuas for idx in layer_indices
                },
                "input_normalization": normalization_kind,
                "subword_pooling": app_config.model.subword_pooling,
            },
            artifacts={"probe_bank": bank_dir, "manifest": manifest_path},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_parsed_corpus_jsonl(path: Path) -> ParsedCorpus:
    """Load a parsed corpus from a JSONL file."""
    import json

    sentences: list[ParsedSentence] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences.append(ParsedSentence.model_validate(json.loads(line)))

    return ParsedCorpus(
        name=path.stem,
        sentences=sentences,
        source=str(path),
    )


def _ensure_cache(
    *,
    cache: ActivationCache,
    split: str,
    sentences: list[ParsedSentence],
    app_config: AppConfig,
) -> list[ParsedSentence]:
    """Extract activations for `sentences` and cache them, unless already cached.

    Returns the (possibly filtered) list of sentences whose activations were
    successfully extracted.
    """
    if cache.has_metadata(split):
        meta = cache.read_metadata(split)
        cached_ids = list(meta["sentence_ids"])  # type: ignore[arg-type]
        sentence_lookup = {s.sentence_id: s for s in sentences}
        if not all(sid in sentence_lookup for sid in cached_ids):
            raise ValueError(
                f"Cache for split {split!r} contains sentence ids not in the parsed corpus. "
                f"You may need to clear the cache directory."
            )
        # Verify pooling consistency: a cache built with one pooling mode is
        # not interchangeable with another. We keep this check soft (warn) so
        # legacy caches without the field still load; new runs always record it.
        cached_pooling = meta.get("extra", {}).get("subword_pooling")
        current_pooling = app_config.model.subword_pooling
        if cached_pooling is not None and cached_pooling != current_pooling:
            raise ValueError(
                f"Cache for split {split!r} was built with subword_pooling="
                f"{cached_pooling!r} but current config uses {current_pooling!r}. "
                f"Clear the cache directory or change the pooling back."
            )

        filtered = [sentence_lookup[sid] for sid in cached_ids]
        logger.info(
            "Using cached activations for %d sentences (split=%s)", len(filtered), split
        )
        return filtered

    # Not cached: extract.
    logger.info("Extracting activations for %d sentences (split=%s)", len(sentences), split)
    extractor = LLMActivationExtractor.from_pretrained(app_config.model)

    items = [(s.sentence_id, s.tokens) for s in sentences]
    extraction = extractor.extract(items)
    successful = {item.sentence_id for item in extraction.items}
    survivors = [s for s in sentences if s.sentence_id in successful]

    if not extraction.items:
        raise RuntimeError(f"No sentences successfully extracted for split={split!r}")

    sample = extraction.items[0]
    num_layers = sample.num_layers()
    hidden_size = sample.hidden_size()

    sentence_ids = [item.sentence_id for item in extraction.items]
    sentence_lengths = [item.num_words() for item in extraction.items]

    cache.write_metadata(
        split=split,
        sentence_ids=sentence_ids,
        sentence_lengths=sentence_lengths,
        hidden_size=hidden_size,
        num_layers=num_layers,
        extra={
            "model_name": app_config.model.model_name,
            "torch_dtype": app_config.model.torch_dtype,
            "subword_pooling": app_config.model.subword_pooling,
            "skipped_sentence_ids": extraction.skipped_sentence_ids,
        },
    )

    for layer_index in range(num_layers):
        per_layer: list[NDArray[np.float32]] = [
            item.activations[layer_index] for item in extraction.items
        ]
        concatenated = np.concatenate(per_layer, axis=0).astype(np.float32)
        cache.write_layer(split=split, layer_index=layer_index, activations=concatenated)

    logger.info(
        "Cached %d sentences across %d layers for split=%s",
        len(extraction.items),
        num_layers,
        split,
    )

    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return survivors


def _ensure_corpus_stats(
    *,
    cache: ActivationCache,
    split: str,
    layer_indices: list[int],
    train_data_per_layer: dict[int, ProbeTrainingData],
) -> None:
    """Compute per-layer corpus mean/std stats if not already present in the cache.

    Stats are deterministic given the cached activations, so we can safely skip
    recomputation when the files exist.
    """
    for layer_index in layer_indices:
        if cache.has_corpus_stats(split=split, layer_index=layer_index):
            continue
        # Build a (n_words, hidden_size) array from the per-sentence activations
        # stored in train_data_per_layer.
        train_data = train_data_per_layer[layer_index]
        all_word_vectors = np.concatenate(train_data.activations, axis=0).astype(np.float32)
        mean, std = compute_corpus_stats(all_word_vectors)
        cache.write_corpus_stats(
            split=split, layer_index=layer_index, mean=mean, std=std
        )
        logger.info(
            "Computed corpus stats for layer %d: mean[0]=%.4f std[0]=%.4f",
            layer_index,
            float(mean[0]),
            float(std[0]),
        )


def _build_layer_data(
    cache: ActivationCache,
    split: str,
    sentences: list[ParsedSentence],
    layer_index: int,
) -> ProbeTrainingData:
    cached = cache.read_split(split, layer_index)
    return build_probe_training_data(parsed_sentences=sentences, cached_split=cached)


# Register at import time
register_experiment_runner("probe_training", ProbeTrainingRunner)


__all__ = ["ProbeTrainingRunner"]
