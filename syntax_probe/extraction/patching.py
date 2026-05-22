"""Activation-patching forward-pass mechanics.

Adds a *hook-based* forward pass to the extraction subsystem: take a
trained LLM, install pre-hooks on a target transformer block, replace
the residual stream at specific (layer, subword) cells with values
cached from a *source* run, and capture the resulting modified hidden
states for downstream probe application.

This is the standard "interchange-intervention" / "activation patching"
protocol from causal-abstraction methodology (Geiger et al. 2023). The
key difference from :class:`LLMActivationExtractor` is that we modify
the residual stream during forward propagation rather than just
recording it. Subsequent transformer blocks therefore compute over the
modified state, and the patched information propagates through
attention to other token positions.

Why a separate extractor rather than extending ``LLMActivationExtractor``:
the responsibilities are distinct enough that mixing them obscures
both. ``LLMActivationExtractor`` is read-only — it captures residual
streams without modification. ``PatchingExtractor`` is read-modify —
it intervenes on the residual stream and captures the result.
Tokenization, batching, alignment, and subword pooling are shared
helpers (imported from ``extractor`` and ``alignment``); only the
forward-pass hook installation differs.

The patching extractor is the engine for the activation-patching
experiment runner; users typically don't call it directly. See
``docs/activation_patching_plan.md`` for protocol details.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from tqdm.auto import tqdm

from ..core.config import ModelConfig
from .alignment import align_words_to_offsets
from .extractor import (
    LayerActivations,
    _model_device,
    _pool_subwords,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch specification dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceCacheKey:
    """Identifies one cell in the source-residual cache.

    The cache stores, for each (sentence_id, layer_index, subword_index)
    triple, a single hidden-state vector. This is the residual stream
    *immediately before* the layer-th transformer block runs (equivalent
    to ``hidden_states[layer_index]`` from a HuggingFace forward pass).
    """

    sentence_id: str
    layer_index: int
    subword_index: int


@dataclass(slots=True)
class PatchSpec:
    """One patching intervention to apply during a single forward pass.

    The hook will replace ``target_residual[batch_index, target_subword_index, :]``
    at layer ``layer_index`` with ``replacement``. The ``layer_index`` is
    interpreted as the input to the ``layer_index``-th transformer block,
    matching the ``hidden_states[layer_index]`` indexing convention.
    """

    batch_index: int
    layer_index: int
    target_subword_index: int
    replacement: torch.Tensor  # shape (hidden_size,), dtype = model dtype


# ---------------------------------------------------------------------------
# Block discovery
# ---------------------------------------------------------------------------


_VISION_PREFIXES: tuple[str, ...] = (
    "vision", "image", "visual", "multi_modal_projector",
    "audio", "speech",
)
"""Submodule name prefixes whose ModuleLists should never be the text
decoder. Used by the recursive fallback in ``discover_transformer_blocks``
to avoid mis-selecting a vision encoder's block list."""


def _expected_text_decoder_depth(model: Any) -> int | None:
    """Read the expected text-decoder ``num_hidden_layers`` from the
    model's config.

    Multimodal HF configs typically nest a ``text_config`` (e.g.,
    ``Gemma3Config.text_config.num_hidden_layers``) so we prefer that
    when present; text-only configs expose the value directly on the
    top-level config.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None
    for outer in ("text_config", "language_config"):
        sub = getattr(config, outer, None)
        if sub is not None:
            n = getattr(sub, "num_hidden_layers", None)
            if isinstance(n, int) and n > 0:
                return n
    n = getattr(config, "num_hidden_layers", None)
    if isinstance(n, int) and n > 0:
        return n
    return None


def _find_module_list_by_length(
    root: Any, expected_n: int, max_depth: int = 5,
) -> tuple[str, torch.nn.ModuleList] | None:
    """BFS for the first ``ModuleList`` of length ``expected_n`` under
    ``root``, skipping vision/audio submodules so we don't pick a vision
    encoder's block list by accident.

    BFS (not DFS) so the *shallowest* matching ModuleList wins — the
    text decoder is reliably shallower than e.g. nested encoder stacks.
    """
    queue: list[tuple[str, Any, int]] = [("", root, 0)]
    while queue:
        path, obj, depth = queue.pop(0)
        if isinstance(obj, torch.nn.ModuleList) and len(obj) == expected_n:
            return path, obj
        if depth >= max_depth:
            continue
        children = getattr(obj, "named_children", None)
        if children is None:
            continue
        for name, child in children():
            if any(name.startswith(p) for p in _VISION_PREFIXES):
                continue
            queue.append((f"{path}.{name}" if path else name, child, depth + 1))
    return None


def discover_transformer_blocks(model: Any) -> torch.nn.ModuleList:
    """Locate the list of transformer blocks for a HuggingFace causal LM.

    Two-stage strategy:

    1. **Explicit paths**: try a list of conventional attribute paths
       that cover all current registry models (Llama/Qwen/Mistral/Gemma
       text-only) and the common HF multimodal text-decoder nesting
       conventions.

    2. **Recursive fallback**: if no explicit path matches, BFS for a
       ``torch.nn.ModuleList`` whose length equals the model's expected
       text-decoder depth, read from
       ``config.text_config.num_hidden_layers`` (multimodal) or
       ``config.num_hidden_layers`` (text-only). Submodules named
       ``vision_*`` / ``image_*`` / ``audio_*`` / ``multi_modal_*`` are
       skipped so the search can't mis-select a vision encoder's blocks.

    The recursive fallback exists because HF has refactored multimodal
    attribute paths multiple times (e.g., Gemma3's text-decoder path
    moved between transformers 4.x and 5.x). When the fallback fires it
    emits a WARNING with the discovered path so it can be promoted to
    an explicit candidate.
    """
    candidates = [
        ("model", "layers"),                       # Llama / Qwen / Mistral / Gemma text-only
        ("model", "language_model", "layers"),     # Gemma3 multimodal (transformers 5.x)
        ("language_model", "model", "layers"),     # Llava-family older style
        ("language_model", "layers"),              # Some VLMs flatten further
        ("transformer", "h"),                      # GPT-2 family
        ("model", "decoder", "layers"),            # OPT
    ]
    for path in candidates:
        obj: Any = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            logger.debug("Discovered transformer blocks at %s (%d blocks)",
                         ".".join(path), len(obj))
            return obj

    # Fallback: search by config-declared depth.
    expected_n = _expected_text_decoder_depth(model)
    if expected_n is not None:
        found = _find_module_list_by_length(model, expected_n)
        if found is not None:
            attr_path, blocks = found
            logger.warning(
                "discover_transformer_blocks: no explicit path matched for "
                "%s; recursive fallback located a %d-block ModuleList at %r. "
                "Consider adding this path to the explicit candidates list.",
                type(model).__name__, len(blocks), attr_path,
            )
            return blocks

    raise RuntimeError(
        f"Could not find transformer-block ModuleList in "
        f"{type(model).__name__}; checked explicit paths {candidates!r} "
        f"and recursive fallback (expected text-decoder depth={expected_n})."
    )


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------


def _make_replace_post_hook(
    patches_by_batch_index: dict[int, list[tuple[int, torch.Tensor]]],
):
    """Build a forward-hook (post-hook) that replaces residual-stream cells
    in a transformer block's output.

    ``patches_by_batch_index`` maps each batch element's index to a list
    of (subword_index, replacement_vector) tuples. The hook clones the
    output tensor and writes the replacements in.

    The returned hook exposes a ``fire_count`` attribute (a single-
    element list of ``int``) that is incremented each time the hook
    runs, so callers can confirm it fired during a forward pass.

    Why a *post*-hook on block ``L_patch − 1`` rather than a pre-hook on
    block ``L_patch``: HuggingFace's ``output_hidden_states`` capture
    appends ``hidden_states`` to ``all_hidden_states`` *before* invoking
    the next block's ``__call__``. A pre-hook on block ``L_patch`` fires
    during that ``__call__`` — too late to affect what was already
    captured as ``hidden_states[L_patch]``. Instead, we hook the
    *output* of block ``L_patch − 1``: its modified output becomes the
    ``hidden_states`` variable that the enclosing forward loop then
    captures as ``hidden_states[L_patch]``, and that block ``L_patch``
    receives as input. Both objectives are met by the same intervention.

    The hook handles both the tuple-output convention (modern decoder
    layers: ``(hidden_states, attention_weights?, past_kv?)``) and the
    plain-tensor convention (for modules like ``nn.Embedding`` if we
    ever extend to ``L_patch == 0``). Raises if it can't find a tensor
    to patch.
    """

    fire_count = [0]

    def post_hook(module, args, output):
        fire_count[0] += 1

        def _patch(x: torch.Tensor) -> torch.Tensor:
            modified = x.clone()
            for batch_idx, replacements in patches_by_batch_index.items():
                for subword_idx, vector in replacements:
                    modified[batch_idx, subword_idx, :] = vector.to(
                        modified.device, dtype=modified.dtype
                    )
            return modified

        if isinstance(output, tuple) and output:
            hidden = output[0]
            if isinstance(hidden, torch.Tensor):
                return (_patch(hidden),) + output[1:]
        if isinstance(output, torch.Tensor):
            return _patch(output)
        raise RuntimeError(
            f"Patching post-hook on {type(module).__name__} could not "
            f"locate hidden_states tensor in the module's output "
            f"(got {type(output).__name__}). The model architecture may "
            f"have changed; update the hook to match."
        )

    post_hook.fire_count = fire_count  # type: ignore[attr-defined]
    return post_hook


@contextmanager
def installed_patches(
    blocks: torch.nn.ModuleList,
    *,
    patches_by_layer_and_batch: dict[int, dict[int, list[tuple[int, torch.Tensor]]]],
):
    """Context manager that installs patch hooks on the appropriate
    blocks and removes them on exit.

    ``patches_by_layer_and_batch`` maps ``L_patch`` →
    (``batch_index`` → list of ``(subword_index, replacement)``).
    ``L_patch`` here uses the *probe-application convention*: a probe
    reads ``hidden_states[L_patch]`` where ``hidden_states[0]`` is the
    embedding output and ``hidden_states[i]`` for ``i ≥ 1`` is the
    output of block ``i − 1``. To make a probe at ``L_patch`` see the
    patched value, we hook the *output* of block ``L_patch − 1``.

    Yields a dict mapping each patched ``L_patch`` to the installed
    hook function. Each hook carries a ``fire_count`` attribute (see
    :func:`_make_replace_post_hook`) so callers can verify it ran.

    ``L_patch = 0`` (patching the embedding output) is not supported:
    it requires hooking the embedding module, which has model-specific
    pre-processing (e.g. Gemma's hidden-size scaling) that the current
    hook doesn't account for. For our use cases all β-peak observational
    layers are ≥ 1, so this restriction has no practical effect.
    """
    handles: list[Any] = []
    hooks_by_layer: dict[int, Any] = {}
    try:
        for layer_index, by_batch in patches_by_layer_and_batch.items():
            if layer_index < 1:
                raise NotImplementedError(
                    f"Patching at layer_index={layer_index} requires "
                    f"hooking the embedding output, which is not "
                    f"supported in this version. Use layer_index ≥ 1."
                )
            if layer_index > len(blocks):
                raise ValueError(
                    f"Patch layer_index={layer_index} out of range. "
                    f"Model has {len(blocks)} blocks; valid range is "
                    f"[1, {len(blocks)}] (1 = hook block 0's output → "
                    f"hidden_states[1])."
                )
            # Hook block (L_patch - 1) with a post-hook. Block (L-1)'s
            # output becomes hidden_states[L] in HF's capture, AND is
            # what block L receives as input.
            target_block = blocks[layer_index - 1]
            hook = _make_replace_post_hook(by_batch)
            # ``prepend=True``: transformers ≥ 5.x captures
            # ``output_hidden_states`` via internal forward hooks
            # installed at model init (the ``@can_record_outputs``
            # mechanism). PyTorch fires forward hooks in registration
            # order, so without ``prepend`` our patch hook would run
            # after those internal hooks and modify the layer output
            # only after it had been recorded into
            # ``outputs.hidden_states[L_patch]``. The probe at
            # ``L_measure = L_patch`` reads from that captured tensor,
            # so the patch must land before the recording hook.
            handle = target_block.register_forward_hook(hook, prepend=True)
            handles.append(handle)
            hooks_by_layer[layer_index] = hook
        yield hooks_by_layer
    finally:
        for handle in handles:
            handle.remove()


# ---------------------------------------------------------------------------
# Patching extractor
# ---------------------------------------------------------------------------


class PatchingExtractor:
    """Run patched forward passes on a target sentence batch.

    Usage pattern:

    1. Load the model and tokenizer once via :meth:`from_pretrained`.
    2. Pre-cache source residuals (the values to splice in) using
       :meth:`cache_source_residuals`, which is a normal forward pass
       that records subword-level activations at the specified
       (sentence, layer, subword) cells.
    3. Call :meth:`extract_with_patches` for each batch of (target,
       source-cache-reference) pairs to produce patched per-word
       activations.

    The :meth:`from_pretrained` helper takes a :class:`ModelConfig`,
    matching :class:`LLMActivationExtractor`'s API.
    """

    def __init__(self, model: Any, tokenizer: Any, config: ModelConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._blocks = discover_transformer_blocks(model)

    @classmethod
    def from_pretrained(cls, config: ModelConfig) -> PatchingExtractor:
        """Mirror :class:`LLMActivationExtractor.from_pretrained`, with
        the tokenizer pinned to right-padding so precomputed standalone
        subword indices index the padded batch tensor at content-token
        positions."""
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
        # Subword indices used by the patching subsystem are computed
        # once from standalone (unpadded) tokenization and reused as
        # positions in the padded batch tensor for both source-cache
        # reads and patch writes. That equivalence holds only with
        # right-padding (content tokens at positions ``[0, len)``);
        # left-padding shifts content tokens rightward by the per-item
        # pad amount.
        if tokenizer.padding_side != "right":
            logger.info(
                "Overriding tokenizer padding_side from %r to 'right' "
                "for PatchingExtractor.", tokenizer.padding_side,
            )
            tokenizer.padding_side = "right"
        # Llama-family tokenizers ship without a pad_token; eos_token
        # is safe here because we only run forward passes and the
        # attention_mask excludes padding from the model's computation.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(
                "Tokenizer had no pad_token; set pad_token = eos_token (%r).",
                tokenizer.eos_token,
            )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            revision=config.revision,
            torch_dtype=getattr(torch, config.torch_dtype),
            trust_remote_code=config.trust_remote_code,
            device_map="auto",
        )
        model.eval()
        return cls(model=model, tokenizer=tokenizer, config=config)

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    # ----- source caching ----------------------------------------------------

    def cache_source_residuals(
        self,
        items: Sequence[tuple[str, list[str]]],
        *,
        cells: dict[str, list[tuple[int, int]]],
        progress: bool = True,
    ) -> dict[SourceCacheKey, torch.Tensor]:
        """Run a normal forward pass on source items and cache residuals
        at specified (layer, subword) cells.

        Args:
            items: list of (sentence_id, words) pairs.
            cells: maps sentence_id to a list of (layer_index, subword_index)
                cells to cache for that sentence. Subword indices are
                *post-tokenization* (including the BOS special token if
                the tokenizer adds one); the caller is responsible for
                computing them via word-to-subword alignment.
            progress: whether to show a progress bar.

        Returns:
            Dict mapping :class:`SourceCacheKey` to the cached vector
            (1-D tensor of length hidden_size, on CPU, dtype=model dtype).
        """
        cache: dict[SourceCacheKey, torch.Tensor] = {}
        batches = list(_batched(items, self.config.batch_size))
        bar: Iterable = (
            tqdm(batches, desc="caching sources", unit="batch")
            if progress else iter(batches)
        )

        for batch in bar:
            sentences = [" ".join(words) for _, words in batch]
            encoded = self.tokenizer(
                sentences,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
            )
            device = _model_device(self.model)
            model_inputs = {
                k: v.to(device) for k, v in encoded.items()
                if isinstance(v, torch.Tensor)
            }
            with torch.no_grad():
                outputs = self.model(
                    **model_inputs, output_hidden_states=True, use_cache=False
                )
            # outputs.hidden_states: tuple of (L+1) tensors, each
            # (batch, seq_len, hidden). hidden_states[layer_index] is the
            # residual *before* block ``layer_index``.
            hidden_states = outputs.hidden_states

            for batch_index, (sentence_id, _words) in enumerate(batch):
                if sentence_id not in cells:
                    continue
                for layer_index, subword_index in cells[sentence_id]:
                    if layer_index >= len(hidden_states):
                        raise IndexError(
                            f"layer_index={layer_index} out of range; "
                            f"model has {len(hidden_states)} hidden-state layers."
                        )
                    vec = hidden_states[layer_index][batch_index, subword_index].detach().cpu()
                    cache[SourceCacheKey(sentence_id, layer_index, subword_index)] = vec
        return cache

    # ----- patched extraction ------------------------------------------------

    def extract_with_patches(
        self,
        items: Sequence[tuple[str, list[str]]],
        *,
        per_item_patches: list[list[PatchSpec] | None],
        progress: bool = False,
    ) -> list[LayerActivations | None]:
        """Run patched forward passes on a sequence of target items.

        Args:
            items: list of (sentence_id, words) pairs (the targets).
            per_item_patches: one entry per item; each entry is either
                ``None`` (no patching for that item — runs as a normal
                forward pass) or a list of :class:`PatchSpec` objects
                describing the residual-stream replacements to apply.
                The ``batch_index`` field of each ``PatchSpec`` is
                ignored on entry; we set it to the item's batch position
                internally.
            progress: whether to show a progress bar.

        Returns:
            A list parallel to ``items``: each entry is either a
            :class:`LayerActivations` (per-word pooled hidden states for
            the patched run) or ``None`` if subword-to-word alignment
            failed for that item.
        """
        if len(items) != len(per_item_patches):
            raise ValueError(
                f"items ({len(items)}) and per_item_patches "
                f"({len(per_item_patches)}) must have equal length."
            )

        results: list[LayerActivations | None] = [None] * len(items)
        batches = list(_batched_with_patches(items, per_item_patches, self.config.batch_size))
        bar: Iterable = (
            tqdm(batches, desc="patching targets", unit="batch")
            if progress else iter(batches)
        )

        for batch_items, batch_patches, original_indices in bar:
            sentences = [" ".join(words) for _, words in batch_items]
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
            model_inputs = {
                k: v.to(device) for k, v in encoded.items()
                if isinstance(v, torch.Tensor)
            }

            # Build per-(layer, batch_index) replacement spec.
            patches_by_layer_and_batch: dict[int, dict[int, list[tuple[int, torch.Tensor]]]] = {}
            for batch_index, patches in enumerate(batch_patches):
                if patches is None:
                    continue
                for spec in patches:
                    by_batch = patches_by_layer_and_batch.setdefault(
                        spec.layer_index, {}
                    )
                    by_batch.setdefault(batch_index, []).append(
                        (spec.target_subword_index, spec.replacement)
                    )

            with torch.no_grad(), installed_patches(
                self._blocks,
                patches_by_layer_and_batch=patches_by_layer_and_batch,
            ) as hooks_by_layer:
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
            # Warn if patches were planned but no hook fired, e.g. for a
            # model whose decoder-layer output type is neither a tuple
            # nor a plain tensor (the two branches the hook handles).
            if hooks_by_layer and all(
                h.fire_count[0] == 0 for h in hooks_by_layer.values()
            ):
                logger.warning(
                    "Patch hooks were installed at layers %s but none "
                    "fired during this batch's forward pass.",
                    sorted(hooks_by_layer),
                )
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            hidden_states_cpu = hidden_states.detach().to(torch.float32).cpu().numpy()

            for batch_index, (sentence_id, words) in enumerate(batch_items):
                offsets_row = offset_mapping[batch_index].tolist()
                offsets = [(int(s), int(e)) for s, e in offsets_row]
                sentence = sentences[batch_index]
                try:
                    alignment = align_words_to_offsets(
                        words=words, offsets=offsets, sentence=sentence
                    )
                except ValueError as err:
                    logger.debug("Alignment failed for %s: %s", sentence_id, err)
                    continue
                per_word = _pool_subwords(
                    hidden_states=hidden_states_cpu,
                    batch_index=batch_index,
                    word_to_subword_indices=alignment.word_to_subword_indices,
                    pooling=self.config.subword_pooling,
                )
                results[original_indices[batch_index]] = LayerActivations(
                    sentence_id=sentence_id,
                    words=list(words),
                    activations=per_word,
                )
        return results


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def _batched_with_patches(
    items: Sequence[tuple[str, list[str]]],
    per_item_patches: Sequence[list[PatchSpec] | None],
    batch_size: int,
) -> Iterable[
    tuple[
        list[tuple[str, list[str]]],
        list[list[PatchSpec] | None],
        list[int],
    ]
]:
    """Yield batches alongside the original index of each item.

    Returning the original index lets the caller place each result back
    into its slot in the output list, even after the patching may have
    re-ordered or padded items.
    """
    for start in range(0, len(items), batch_size):
        end = min(start + batch_size, len(items))
        yield (
            list(items[start:end]),
            list(per_item_patches[start:end]),
            list(range(start, end)),
        )


__all__ = [
    "PatchSpec",
    "PatchingExtractor",
    "SourceCacheKey",
    "discover_transformer_blocks",
    "installed_patches",
]
