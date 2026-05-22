"""On-disk activation cache.

Activations are stored as one ``.safetensors`` file per
``(corpus, model, layer)`` tuple, holding a single ``bfloat16`` tensor of
shape ``(total_words, hidden_size)``. The companion ``meta.{split}.json``
records sentence ids and per-sentence word counts; reading it does not
require touching any layer file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load_file as _safetensors_load_file
from safetensors.torch import save_file as _safetensors_save_file

logger = logging.getLogger(__name__)


# Suffixes used for layer files. The new format takes priority on read.
_LAYER_SUFFIX = ".safetensors"
_LAYER_LEGACY_SUFFIX = ".npz"

# Set once per process the first time we hit a legacy file, so the warning
# doesn't spam the log on a 32-layer model.
_legacy_warning_emitted = False


@dataclass(frozen=True, slots=True)
class ActivationCacheKey:
    """Identifies a (corpus, model) pair. One cache directory per key."""

    corpus_name: str
    model_name: str
    cache_root: Path

    def directory(self) -> Path:
        # Sanitize the model name for use as a path component.
        safe_model = self.model_name.replace("/", "__")
        return self.cache_root / safe_model / self.corpus_name


@dataclass(frozen=True, slots=True)
class CachedSplit:
    """Per-split cache content loaded from disk."""

    sentence_ids: list[str]
    sentence_lengths: NDArray[np.int32]
    """Number of words per sentence."""

    activations: NDArray[np.float32]
    """Concatenated word activations, shape (total_words, hidden_size).
    Always returned in float32 for downstream probe code, regardless of
    the on-disk dtype."""

    def num_sentences(self) -> int:
        return len(self.sentence_ids)

    def slice_for_sentence(self, sentence_index: int) -> NDArray[np.float32]:
        """Return the activation slice for one sentence, shape (n_words, hidden_size)."""
        starts = np.concatenate([[0], np.cumsum(self.sentence_lengths)])
        return self.activations[starts[sentence_index] : starts[sentence_index + 1]]


class ActivationCache:
    """Read/write per-layer activation files.

    Each layer's activations are stored as
    ``{split}.layer_{idx:03d}.safetensors`` inside the cache directory.
    The ``meta.{split}.json`` file records the sentence ids and
    per-sentence word counts; reading it doesn't require reading any
    layer file.
    """

    def __init__(self, key: ActivationCacheKey) -> None:
        self.key = key
        self.directory = key.directory()
        self.directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _layer_path(self, split: str, layer_index: int) -> Path:
        return self.directory / f"{split}.layer_{layer_index:03d}{_LAYER_SUFFIX}"

    def _legacy_layer_path(self, split: str, layer_index: int) -> Path:
        return self.directory / f"{split}.layer_{layer_index:03d}{_LAYER_LEGACY_SUFFIX}"

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def write_metadata(
        self,
        *,
        split: str,
        sentence_ids: list[str],
        sentence_lengths: list[int],
        hidden_size: int,
        num_layers: int,
        extra: dict[str, object] | None = None,
    ) -> None:
        meta_path = self.directory / f"meta.{split}.json"
        payload = {
            "split": split,
            "sentence_ids": sentence_ids,
            "sentence_lengths": sentence_lengths,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "extra": extra or {},
        }
        meta_path.write_text(json.dumps(payload), encoding="utf-8")

    def read_metadata(self, split: str) -> dict[str, object]:
        meta_path = self.directory / f"meta.{split}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No cache metadata found at {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def has_metadata(self, split: str) -> bool:
        return (self.directory / f"meta.{split}.json").exists()

    def invalidate(self, split: str) -> None:
        """Remove all cached files for ``split`` so the next call re-extracts.

        Removes ``meta.{split}.json``, all current-format
        ``{split}.layer_*.safetensors`` files, and any legacy
        ``{split}.layer_*.npz`` files. Safe to call even when no cache
        exists.
        """
        meta_path = self.directory / f"meta.{split}.json"
        if meta_path.exists():
            meta_path.unlink()
        for suffix in (_LAYER_SUFFIX, _LAYER_LEGACY_SUFFIX):
            for layer_file in self.directory.glob(f"{split}.layer_*{suffix}"):
                layer_file.unlink()

    # ------------------------------------------------------------------
    # Per-layer activations
    # ------------------------------------------------------------------

    def write_layer(
        self,
        *,
        split: str,
        layer_index: int,
        activations: NDArray[np.float32],
    ) -> Path:
        """Write one layer's activations to ``.safetensors`` as bfloat16.

        The input is expected to be a contiguous float32 numpy array
        (shape ``(total_words, hidden_size)``). It's converted to a bf16
        torch tensor (no copy of the float32 buffer; the cast allocates
        the bf16 buffer once) and saved.

        The float32->bf16 cast uses round-to-nearest-even and is bit-exact
        for values that originated as bf16 from the LLM forward pass.
        """
        path = self._layer_path(split, layer_index)
        # ``torch.from_numpy`` shares memory with the numpy buffer; the
        # subsequent ``.to(torch.bfloat16)`` allocates the bf16 destination
        # once and copies. Peak memory during this step is ~1.5x the
        # float32 size (float32 source + bf16 dest), which is fine for
        # the layer sizes we deal with (~1.7 GB float32 -> ~860 MB bf16).
        tensor_bf16 = torch.from_numpy(activations).to(torch.bfloat16)
        _safetensors_save_file({"activations": tensor_bf16}, str(path))
        return path

    def read_layer(self, *, split: str, layer_index: int) -> NDArray[np.float32]:
        """Load one layer's activations as float32 numpy.

        Tries the current ``.safetensors`` format first; falls back to
        the legacy ``.npz`` format if only a legacy file is present.
        Always returns float32 regardless of the on-disk dtype.
        """
        path = self._layer_path(split, layer_index)
        if path.exists():
            loaded = _safetensors_load_file(str(path))["activations"]
            # bf16 -> float32 -> numpy. The .float() upcast is lossless
            # and produces a contiguous tensor that .numpy() can view
            # without further copy.
            return loaded.float().numpy()

        legacy_path = self._legacy_layer_path(split, layer_index)
        if legacy_path.exists():
            global _legacy_warning_emitted
            if not _legacy_warning_emitted:
                logger.warning(
                    "Reading legacy compressed-npz activation cache at %s. "
                    "Re-extract this corpus/model to switch to the faster "
                    "safetensors+bf16 format (see invalidate()).",
                    legacy_path.parent,
                )
                _legacy_warning_emitted = True
            with np.load(legacy_path) as data:
                return data["activations"].astype(np.float32)

        raise FileNotFoundError(
            f"No cached activations for split={split!r} layer={layer_index} "
            f"under {self.directory} (looked for {path.name} and {legacy_path.name})."
        )

    def read_split(self, split: str, layer_index: int) -> CachedSplit:
        meta = self.read_metadata(split)
        return CachedSplit(
            sentence_ids=list(meta["sentence_ids"]),
            sentence_lengths=np.asarray(meta["sentence_lengths"], dtype=np.int32),
            activations=self.read_layer(split=split, layer_index=layer_index),
        )

    def has_layer(self, *, split: str, layer_index: int) -> bool:
        """True if either the current or legacy layer file exists."""
        return (
            self._layer_path(split, layer_index).exists()
            or self._legacy_layer_path(split, layer_index).exists()
        )

    def num_layers(self, split: str) -> int:
        meta = self.read_metadata(split)
        return int(meta["num_layers"])  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Per-layer corpus statistics for probe input normalization
    # ------------------------------------------------------------------
    #
    # When ``ProbeConfig.input_normalization == "per_corpus_standardize"``,
    # the probe needs per-dimension mean and std computed once over the
    # training split. We store these alongside the activation files so
    # they're computed once and shared across multiple probe-training
    # runs that use the same cache.
    #
    # These tensors are tiny (one hidden_size-vector each), so we keep
    # them in numpy float32 ``.npz`` form — no migration value.

    def corpus_stats_path(self, *, split: str, layer_index: int) -> Path:
        return self.directory / f"{split}.layer_{layer_index:03d}.stats.npz"

    def has_corpus_stats(self, *, split: str, layer_index: int) -> bool:
        return self.corpus_stats_path(split=split, layer_index=layer_index).exists()

    def write_corpus_stats(
        self,
        *,
        split: str,
        layer_index: int,
        mean: NDArray[np.float32],
        std: NDArray[np.float32],
    ) -> Path:
        path = self.corpus_stats_path(split=split, layer_index=layer_index)
        np.savez(
            path,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
        )
        return path

    def read_corpus_stats(
        self, *, split: str, layer_index: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        path = self.corpus_stats_path(split=split, layer_index=layer_index)
        if not path.exists():
            raise FileNotFoundError(
                f"No corpus stats found at {path}. Run probe training with "
                f"input_normalization='per_corpus_standardize' to compute them."
            )
        with np.load(path) as data:
            mean = data["mean"].astype(np.float32)
            std = data["std"].astype(np.float32)
        return mean, std


def make_cache_key(*, corpus_name: str, model_name: str, cache_root: Path) -> ActivationCacheKey:
    return ActivationCacheKey(corpus_name=corpus_name, model_name=model_name, cache_root=cache_root)


def cache_signature(*, corpus_name: str, model_name: str, extras: dict[str, object]) -> str:
    """Stable short signature combining corpus, model, and any extras."""
    payload = json.dumps(
        {"corpus": corpus_name, "model": model_name, "extras": extras},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "ActivationCache",
    "ActivationCacheKey",
    "CachedSplit",
    "cache_signature",
    "make_cache_key",
]
