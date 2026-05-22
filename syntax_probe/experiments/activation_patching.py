"""Activation-patching experiment runner.

Orchestrates the Tier-1 causal experiments (W2, W4, N1) defined in
``docs/activation_patching_plan.md``. The flow per run:

1. Resolve and load a trained probe bank for the model (mirroring
   :class:`apply_probes.ApplyProbesRunner`).
2. Generate the underlying observational stimuli (wh-extraction or
   c-command).
3. Generate variant-specific trial specs from
   :mod:`syntax_probe.stimuli.patching_pairs`.
4. Pre-compute word-to-subword alignments for every source and target
   stimulus involved in the trials, so that intervention-position
   subword indices are known before any model forward pass.
5. Cache source-residual cells via
   :class:`PatchingExtractor.cache_source_residuals`.
6. Run a baseline (unpatched) forward pass on every unique target
   stimulus and apply the probe to record per-pair, per-layer
   distances.
7. Run patched forward passes — one per trial — splicing in the
   cached source residual at the trial's intervention cell.
8. Apply the probe to each patched target run at the trial's
   ``measurement_layers``; record per-pair distances.
9. Write a per-trial JSONL containing both patched and unpatched
   distances at the relevant cells, plus a manifest.

Δβ statistics are computed downstream by the stats pipeline; this
runner concerns itself only with raw measurement.

Why a separate runner rather than extending ``apply_probes``: the
control flow is fundamentally different. ``apply_probes`` runs one
extraction over a stimulus set and applies the probe at every layer;
the patching runner orchestrates pairs of extractions (source +
target) for each trial, each with a hook-installed forward pass.
Mixing the two would obscure both.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ..core.config import (
    ActivationPatchingConfig,
    AppConfig,
    N1VariantConfig,
    W2VariantConfig,
    W4VariantConfig,
)
from ..core.context import RunContext, write_manifest
from ..core.io import write_json, write_jsonl
from ..core.seed import seed_everything
from ..extraction.alignment import WordAlignment, align_words_to_offsets
from ..extraction.patching import (
    PatchingExtractor,
    PatchSpec,
    SourceCacheKey,
)
from ..probes.training import LayerProbeBank
from ..stimuli.c_command import generate_c_command_stimuli
from ..stimuli.patching_pairs import (
    TrialSpec,
    generate_n1_trials,
    generate_w2_trials,
    generate_w4_trials,
)
from ..stimuli.schema import StimulusRecord
from ..stimuli.wh_extraction import generate_wh_extraction_stimuli
from .apply_probes import (
    _apply_probe_to_pairs,
    _check_pooling_consistency,
    _resolve_probe_run_dir,
)
from .base import ExperimentResult, ExperimentRunner
from .registry import register_experiment_runner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ActivationPatchingRunner(ExperimentRunner):
    kind = "activation_patching"

    def run(
        self, *, app_config: AppConfig, context: RunContext
    ) -> ExperimentResult:
        if not isinstance(app_config.experiment, ActivationPatchingConfig):
            raise TypeError("Expected activation_patching config")

        seed_everything(app_config.runtime.seed)
        cfg = app_config.experiment

        # --- 1. Probe bank --------------------------------------------------
        # We can reuse apply_probes' resolver because ActivationPatchingConfig
        # exposes the same probe_run_dir / probe_run_name fields and the
        # resolver only reads those two attributes.
        probe_run_dir = _resolve_probe_run_dir(
            cfg, output_dir=app_config.runtime.output_dir  # type: ignore[arg-type]
        )
        logger.info("Using probe run dir: %s", probe_run_dir)
        _check_pooling_consistency(probe_run_dir, app_config.model.subword_pooling)

        probe_bank_dir = probe_run_dir / "artifacts" / "probe_bank"
        bank = LayerProbeBank.load(probe_bank_dir)
        logger.info(
            "Loaded probe bank: %d layers, rank=%d, hidden_size=%d",
            len(bank.probes), bank.rank, bank.hidden_size,
        )

        # Move probes to the inference device once, up-front. Both the
        # unpatched-baseline pass (step 6) and the per-trial probe
        # application (step 8) call ``_apply_probe_to_pairs``, which
        # moves the input tensor to ``device`` and runs ``probe(vectors)``;
        # if probes haven't been moved here, that ``probe(vectors)`` mm
        # raises a device-mismatch RuntimeError.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for probe in bank.probes.values():
            probe.to(device)

        # --- 2. Stimuli -----------------------------------------------------
        if cfg.stimuli_kind == "wh_extraction":
            stimulus_set = generate_wh_extraction_stimuli(
                num_items=cfg.stimuli_config.num_items
            )
        elif cfg.stimuli_kind == "c_command":
            stimulus_set = generate_c_command_stimuli(
                num_items=cfg.stimuli_config.num_items
            )
        else:
            raise NotImplementedError(f"stimuli_kind={cfg.stimuli_kind!r}")
        logger.info("Generated %d stimuli", len(stimulus_set.records))

        # --- 3. Trials ------------------------------------------------------
        trials = _generate_trials(cfg, stimulus_set)
        logger.info("Generated %d trials (variant=%s)",
                    len(trials), cfg.variant_config.variant)
        if not trials:
            raise RuntimeError(
                f"No trials generated for variant={cfg.variant_config.variant!r}. "
                f"Check that stimuli contain the expected conditions and "
                f"role_indices."
            )

        # --- 4. Pre-compute alignments for all involved stimuli -------------
        records_by_id = {r.stimulus_id: r for r in stimulus_set.records}
        involved_stim_ids: set[str] = set()
        for trial in trials:
            involved_stim_ids.add(trial.source_stimulus_id)
            involved_stim_ids.add(trial.target_stimulus_id)
        involved_records = [records_by_id[sid] for sid in involved_stim_ids
                            if sid in records_by_id]
        logger.info(
            "Pre-computing alignments for %d unique stimuli (sources + targets)",
            len(involved_records),
        )

        extractor = PatchingExtractor.from_pretrained(app_config.model)
        alignments = _compute_alignments(
            extractor=extractor,
            records=involved_records,
            max_length=app_config.model.max_length,
        )
        skipped_alignment = [
            sid for sid in involved_stim_ids if sid not in alignments
        ]
        if skipped_alignment:
            logger.warning(
                "Skipped %d stimuli due to alignment failures (will skip "
                "trials referencing them)", len(skipped_alignment),
            )

        # --- 5. Source cache plan -------------------------------------------
        cache_cells = _build_source_cache_plan(
            trials=trials, records_by_id=records_by_id, alignments=alignments,
        )
        # source_items: list of (stim_id, words) for each source needed.
        source_items = [
            (records_by_id[sid].stimulus_id, list(records_by_id[sid].tokens))
            for sid in cache_cells
        ]
        logger.info(
            "Caching source residuals: %d unique source stimuli × "
            "(layer, subword) cells", len(source_items),
        )
        source_cache = extractor.cache_source_residuals(
            source_items, cells=cache_cells, progress=True,
        )

        # --- 6. Unpatched baseline ------------------------------------------
        # For each unique target stimulus, run an unpatched forward pass and
        # apply probes at the union of measurement_layers across trials that
        # use this target.
        unique_target_ids = sorted({t.target_stimulus_id for t in trials})
        unique_target_records = [
            records_by_id[sid] for sid in unique_target_ids
            if sid in alignments
        ]
        logger.info(
            "Running unpatched baseline on %d unique targets",
            len(unique_target_records),
        )
        unpatched_distances = _run_unpatched_baseline(
            extractor=extractor,
            records=unique_target_records,
            bank=bank,
            trials=trials,
        )

        # --- 7. Patched forward passes (one per trial) ----------------------
        executable_trials = [
            t for t in trials
            if t.source_stimulus_id in alignments
            and t.target_stimulus_id in alignments
        ]
        n_skipped_trials = len(trials) - len(executable_trials)
        if n_skipped_trials:
            logger.warning(
                "Skipping %d trials whose source or target failed alignment",
                n_skipped_trials,
            )

        target_items_for_trials, per_item_patches = _build_patched_targets(
            trials=executable_trials,
            records_by_id=records_by_id,
            alignments=alignments,
            source_cache=source_cache,
        )
        logger.info("Running %d patched target forward passes",
                    len(executable_trials))
        patched_results = extractor.extract_with_patches(
            target_items_for_trials,
            per_item_patches=per_item_patches,
            progress=True,
        )

        # --- 8. Apply probes + assemble per-trial rows ----------------------
        # Probes were moved to ``device`` up-front (after step 1).
        per_trial_rows: list[dict[str, Any]] = []
        n_skipped_inference = 0
        for trial, patched_result in zip(executable_trials, patched_results):
            if patched_result is None:
                n_skipped_inference += 1
                continue
            target_record = records_by_id[trial.target_stimulus_id]
            patched_dists = _apply_probe_at_layers(
                bank=bank,
                activations=patched_result.activations,
                record=target_record,
                pairs=trial.measurement_pairs,
                layers=trial.measurement_layers,
                device=device,
            )
            unpatched_for_target = unpatched_distances.get(
                trial.target_stimulus_id, {}
            )
            unpatched_dists = _subset_distances(
                unpatched_for_target, trial.measurement_pairs,
                trial.measurement_layers,
            )
            per_trial_rows.append({
                "trial_id": trial.trial_id,
                "experiment_kind": trial.experiment_kind,
                "item_id": trial.item_id,
                "source_stimulus_id": trial.source_stimulus_id,
                "target_stimulus_id": trial.target_stimulus_id,
                "intervention_role": trial.intervention_role,
                "intervention_layer": trial.intervention_layer,
                "measurement_layers": list(trial.measurement_layers),
                "metadata": dict(trial.metadata),
                "patched_distances": patched_dists,
                "unpatched_distances": unpatched_dists,
            })
        if n_skipped_inference:
            logger.warning(
                "Skipped %d trials whose patched forward pass failed alignment",
                n_skipped_inference,
            )

        # --- 9. Outputs -----------------------------------------------------
        per_trial_path = context.artifact_dir / "per_trial_predictions.jsonl"
        write_jsonl(per_trial_path, iter(per_trial_rows))

        unpatched_path = context.artifact_dir / "unpatched_baseline.json"
        write_json(unpatched_path, unpatched_distances)

        manifest_path = write_manifest(
            context,
            {
                "probe_run_dir": str(probe_run_dir),
                "input_normalization": bank.input_normalization,
                "subword_pooling": app_config.model.subword_pooling,
                "stimuli_kind": cfg.stimuli_kind,
                "variant": cfg.variant_config.variant,
                "stimulus_count": len(stimulus_set.records),
                "trial_count_planned": len(trials),
                "trial_count_executed": len(per_trial_rows),
                "alignment_failures": len(skipped_alignment),
                "artifacts": {
                    "per_trial_predictions": str(per_trial_path),
                    "unpatched_baseline": str(unpatched_path),
                },
            },
        )

        return ExperimentResult(
            experiment_kind=self.kind,
            run_id=context.run_id,
            summary={
                "variant": cfg.variant_config.variant,
                "trial_count_planned": len(trials),
                "trial_count_executed": len(per_trial_rows),
            },
            artifacts={
                "per_trial_predictions": per_trial_path,
                "unpatched_baseline": unpatched_path,
                "manifest": manifest_path,
            },
        )


# ---------------------------------------------------------------------------
# Trial generation dispatch
# ---------------------------------------------------------------------------


def _generate_trials(
    cfg: ActivationPatchingConfig, stimulus_set: Any
) -> list[TrialSpec]:
    v = cfg.variant_config
    if isinstance(v, W2VariantConfig):
        return generate_w2_trials(
            stimulus_set=stimulus_set,
            intervention_layer_per_model=v.intervention_layer,
            intervention_roles=tuple(v.intervention_roles),
            source_target_pairs=tuple(tuple(p) for p in v.source_target_pairs),
            filter_to_lexical_dp=v.filter_to_lexical_dp,
        )
    if isinstance(v, W4VariantConfig):
        return generate_w4_trials(
            stimulus_set=stimulus_set,
            intervention_layers=tuple(v.intervention_layers),
            measurement_layer=v.measurement_layer,
            intervention_role=v.intervention_role,
            source_target_pair=tuple(v.source_target_pair),
            filter_to_lexical_dp=v.filter_to_lexical_dp,
        )
    if isinstance(v, N1VariantConfig):
        return generate_n1_trials(
            stimulus_set=stimulus_set,
            intervention_layer_per_model=v.intervention_layer,
            bidirectional=v.bidirectional,
        )
    raise NotImplementedError(f"Unknown variant: {v!r}")


# ---------------------------------------------------------------------------
# Alignment computation (no model forward pass needed)
# ---------------------------------------------------------------------------


def _compute_alignments(
    *,
    extractor: PatchingExtractor,
    records: Sequence[StimulusRecord],
    max_length: int,
) -> dict[str, WordAlignment]:
    """Tokenize each stimulus and compute its word-to-subword alignment.

    We need this *before* any forward pass so that per-trial patch
    specifications can carry concrete subword indices. The compute is
    cheap (tokenization only, no model run) and runs once per unique
    stimulus.
    """
    out: dict[str, WordAlignment] = {}
    tokenizer = extractor.tokenizer
    for record in tqdm(records, desc="aligning", unit="stim"):
        sentence = " ".join(record.tokens)
        encoded = tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        offsets_row = encoded["offset_mapping"][0].tolist()
        offsets = [(int(s), int(e)) for s, e in offsets_row]
        try:
            alignment = align_words_to_offsets(
                words=list(record.tokens),
                offsets=offsets,
                sentence=sentence,
            )
        except ValueError as err:
            logger.debug("Alignment failed for %s: %s",
                         record.stimulus_id, err)
            continue
        out[record.stimulus_id] = alignment
    return out


def _first_subword_index_for_role(
    *,
    record: StimulusRecord,
    role: str,
    alignment: WordAlignment,
) -> int | None:
    """Return the subword index of the role's first word, or None if the
    role is missing from this record's role_indices."""
    if role not in record.role_indices:
        return None
    word_idx = record.role_indices[role]
    subword_groups = alignment.word_to_subword_indices
    if word_idx >= len(subword_groups):
        return None
    group = subword_groups[word_idx]
    if not group:
        return None
    return int(group[0])


# ---------------------------------------------------------------------------
# Source cache planning
# ---------------------------------------------------------------------------


def _build_source_cache_plan(
    *,
    trials: Sequence[TrialSpec],
    records_by_id: dict[str, StimulusRecord],
    alignments: dict[str, WordAlignment],
) -> dict[str, list[tuple[int, int]]]:
    """Compute the (layer, subword_idx) cells we need to cache from
    each source stimulus, deduplicated.

    Returns:
        Dict mapping source_stimulus_id to a list of (layer, subword_idx)
        cells to cache from that source's forward pass.
    """
    plan: dict[str, set[tuple[int, int]]] = {}
    for trial in trials:
        if trial.source_stimulus_id not in alignments:
            continue
        source_record = records_by_id.get(trial.source_stimulus_id)
        if source_record is None:
            continue
        alignment = alignments[trial.source_stimulus_id]
        subword_idx = _first_subword_index_for_role(
            record=source_record,
            role=trial.intervention_role,
            alignment=alignment,
        )
        if subword_idx is None:
            logger.debug(
                "Cannot resolve subword for role %r in source %s",
                trial.intervention_role, trial.source_stimulus_id,
            )
            continue
        plan.setdefault(trial.source_stimulus_id, set()).add(
            (trial.intervention_layer, subword_idx)
        )
    return {sid: sorted(cells) for sid, cells in plan.items()}


# ---------------------------------------------------------------------------
# Patched-target construction
# ---------------------------------------------------------------------------


def _build_patched_targets(
    *,
    trials: Sequence[TrialSpec],
    records_by_id: dict[str, StimulusRecord],
    alignments: dict[str, WordAlignment],
    source_cache: dict[SourceCacheKey, torch.Tensor],
) -> tuple[list[tuple[str, list[str]]], list[list[PatchSpec] | None]]:
    """For each trial, build (target_item, [patch_spec]) ready for
    :meth:`PatchingExtractor.extract_with_patches`.

    Note: target_items has one entry per trial (not per unique target),
    because two trials sharing a target but differing in patch must run
    as separate forward passes. The extractor batches them together
    where possible — each batch element gets its own PatchSpec.
    """
    items: list[tuple[str, list[str]]] = []
    patches: list[list[PatchSpec] | None] = []
    skipped_no_cache = 0
    skipped_no_subword = 0

    for trial in trials:
        target_record = records_by_id[trial.target_stimulus_id]
        target_alignment = alignments[trial.target_stimulus_id]
        target_subword = _first_subword_index_for_role(
            record=target_record,
            role=trial.intervention_role,
            alignment=target_alignment,
        )
        source_record = records_by_id[trial.source_stimulus_id]
        source_alignment = alignments[trial.source_stimulus_id]
        source_subword = _first_subword_index_for_role(
            record=source_record,
            role=trial.intervention_role,
            alignment=source_alignment,
        )
        if target_subword is None or source_subword is None:
            skipped_no_subword += 1
            continue
        cache_key = SourceCacheKey(
            sentence_id=trial.source_stimulus_id,
            layer_index=trial.intervention_layer,
            subword_index=source_subword,
        )
        replacement = source_cache.get(cache_key)
        if replacement is None:
            skipped_no_cache += 1
            continue

        items.append(
            (
                # Make the per-trial item id unique even when multiple trials
                # share a target stimulus — extract_with_patches uses the id
                # only for logging, but uniqueness avoids confusion downstream.
                f"{trial.trial_id}|target",
                list(target_record.tokens),
            )
        )
        patches.append([
            PatchSpec(
                batch_index=-1,  # set by the extractor per-batch
                layer_index=trial.intervention_layer,
                target_subword_index=target_subword,
                replacement=replacement,
            )
        ])

    if skipped_no_cache or skipped_no_subword:
        logger.warning(
            "Patched-target construction skipped %d trials "
            "(missing cache: %d, missing subword: %d)",
            skipped_no_cache + skipped_no_subword,
            skipped_no_cache, skipped_no_subword,
        )
    return items, patches


# ---------------------------------------------------------------------------
# Probe application
# ---------------------------------------------------------------------------


def _apply_probe_at_layers(
    *,
    bank: LayerProbeBank,
    activations: NDArray[np.float32],
    record: StimulusRecord,
    pairs: Sequence[tuple[str, str, str]],
    layers: Sequence[int],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Apply the per-layer probe at each requested layer; return a
    nested dict ``{pair_label: {str(layer): distance}}``.

    JSON requires string keys for dicts, so layer indices are stringified.
    """
    out: dict[str, dict[str, float]] = {}
    for layer_index in layers:
        if layer_index >= activations.shape[0]:
            continue
        probe = bank.probes.get(layer_index)
        if probe is None:
            continue
        layer_vectors = activations[layer_index]  # (n_words, hidden_size)
        pair_results = _apply_probe_to_pairs(
            probe=probe,
            layer_vectors=layer_vectors,
            record=record,
            word_pairs=pairs,
            device=device,
        )
        for pair_label, distance in pair_results.items():
            out.setdefault(pair_label, {})[str(layer_index)] = distance
    return out


def _run_unpatched_baseline(
    *,
    extractor: PatchingExtractor,
    records: Sequence[StimulusRecord],
    bank: LayerProbeBank,
    trials: Sequence[TrialSpec],
) -> dict[str, dict[str, dict[str, float]]]:
    """Run unpatched forward passes on each unique target stimulus and
    record per-pair, per-layer probe distances at the union of
    measurement layers and pairs needed by the trial set.

    Returns a dict mapping target_stimulus_id to
    ``{pair_label: {str(layer): distance}}``.
    """
    if not records:
        return {}
    items = [(r.stimulus_id, list(r.tokens)) for r in records]
    none_patches: list[list[PatchSpec] | None] = [None] * len(items)
    activations = extractor.extract_with_patches(
        items, per_item_patches=none_patches, progress=True,
    )

    # Union of (pair_label) across trials, and union of measurement_layers.
    pair_set: set[tuple[str, str, str]] = set()
    layer_set: set[int] = set()
    for trial in trials:
        for p in trial.measurement_pairs:
            pair_set.add(p)
        for L in trial.measurement_layers:
            layer_set.add(L)
    pairs = sorted(pair_set, key=lambda p: p[2])
    layers = sorted(layer_set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: dict[str, dict[str, dict[str, float]]] = {}
    for record, result in zip(records, activations):
        if result is None:
            continue
        out[record.stimulus_id] = _apply_probe_at_layers(
            bank=bank,
            activations=result.activations,
            record=record,
            pairs=pairs,
            layers=layers,
            device=device,
        )
    return out


def _subset_distances(
    distances: dict[str, dict[str, float]],
    pairs: Sequence[tuple[str, str, str]],
    layers: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Restrict a {pair: {layer: dist}} dict to the requested pairs and
    layers."""
    pair_labels = {p[2] for p in pairs}
    layer_keys = {str(L) for L in layers}
    return {
        pair_label: {
            L: dist for L, dist in by_layer.items() if L in layer_keys
        }
        for pair_label, by_layer in distances.items()
        if pair_label in pair_labels
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_experiment_runner("activation_patching", ActivationPatchingRunner)


__all__ = ["ActivationPatchingRunner"]
