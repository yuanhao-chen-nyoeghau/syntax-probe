"""Apply-probes runner.

Takes a trained `LayerProbeBank` and a set of experimental stimuli, runs the
LLM on the stimuli, applies each layer's probe to predict pairwise distances,
and aggregates per-condition predicted distances at every layer.

The output is the layer profile for the experiment: for each (layer, word-pair,
condition) tuple, the mean predicted distance and a per-stimulus dump for
downstream statistical analysis.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ..core.config import AppConfig, ApplyProbesConfig
from ..core.context import RunContext, write_manifest
from ..core.io import write_json, write_jsonl
from ..core.seed import seed_everything
from ..extraction.cache import ActivationCache, make_cache_key
from ..extraction.extractor import LLMActivationExtractor
from ..probes.structural import StructuralProbe
from ..probes.training import LayerProbeBank
from ..stimuli.c_command import generate_c_command_stimuli
from ..stimuli.schema import StimulusRecord
from ..stimuli.wh_extraction import generate_wh_extraction_stimuli
from .base import ExperimentResult, ExperimentRunner
from .registry import register_experiment_runner

logger = logging.getLogger(__name__)


# Word pairs we measure per experiment. Each pair is (left_role, right_role, label).
#
# For experiments with multiple sub-experiments (currently only ``c_command``),
# the value is a dict from sub-experiment name to its pair list, and the
# dispatch happens per-record using ``record.metadata["subexperiment"]``.
_PAIRS_BY_EXPERIMENT: dict[str, list[tuple[str, str, str]]] = {
    "wh_extraction": [
        # ``evb`` = embedded verb. We avoid ``embed`` here because the term is
        # heavily overloaded in NLP (vector embeddings); using ``evb`` keeps
        # plot titles, log lines, and paper figures unambiguous.
        ("wh", "embedded_verb", "wh-evb"),
        ("wh", "embedded_subject", "wh-esubj"),
        ("embedded_subject", "embedded_verb", "esubj-evb"),
    ],
}

# Sub-experiment-specific pair tables for c_command. Looked up per-record
# using ``record.metadata["subexperiment"]``.
_C_COMMAND_PAIRS_BY_SUBEXPERIMENT: dict[str, list[tuple[str, str, str]]] = {
    # Reflexive binding: the within-stimulus contrast d(subject, anaphor) vs.
    # d(modifier, anaphor) tests c-command sensitivity; the reflexive vs.
    # pronoun and gender match vs. swap manipulations isolate UD-confounded
    # signal from binding-relevant signal.
    "reflexive": [
        ("subject", "anaphor", "subj-anaphor"),
        ("modifier", "anaphor", "mod-anaphor"),
    ],
    # Principle C: distance between cataphoric pronoun and the R-expression.
    # Differs between violation (pronoun c-commands) and obviated (pronoun in
    # adjunct, doesn't c-command).
    "principle_c": [
        ("pronoun", "r_expression", "pron-rexp"),
    ],
    # Bound-variable anaphora: distance between the quantifier-headed NP and
    # the pronoun. Tests SS vs. LF c-command (the latter via QR).
    "bound_var": [
        ("quant_noun", "pronoun", "qnoun-pron"),
        ("quantifier", "pronoun", "quant-pron"),
    ],
}


def _word_pairs_for_record(
    stimuli_kind: str, record: StimulusRecord,
) -> list[tuple[str, str, str]]:
    """Return the word pairs to measure for a single stimulus record.

    Most experiments have one fixed pair table per ``stimuli_kind``. The
    ``c_command`` experiment dispatches per record using its sub-experiment
    metadata.
    """
    if stimuli_kind == "c_command":
        sub = record.metadata.get("subexperiment")
        if sub is None:
            raise ValueError(
                f"c_command record {record.stimulus_id!r} is missing the "
                f"'subexperiment' metadata field."
            )
        return _C_COMMAND_PAIRS_BY_SUBEXPERIMENT[sub]
    return _PAIRS_BY_EXPERIMENT[stimuli_kind]


class ApplyProbesRunner(ExperimentRunner):
    kind = "apply_probes"

    def run(self, *, app_config: AppConfig, context: RunContext) -> ExperimentResult:
        if not isinstance(app_config.experiment, ApplyProbesConfig):
            raise TypeError("Expected apply_probes config")

        seed_everything(app_config.runtime.seed)
        cfg = app_config.experiment

        # 0. Resolve probe_run_dir (either explicit, or auto-resolve from probe_run_name).
        probe_run_dir = _resolve_probe_run_dir(
            cfg, output_dir=app_config.runtime.output_dir
        )
        logger.info("Using probe run dir: %s", probe_run_dir)

        # 0b. Verify the training run's pooling matches the current model config.
        _check_pooling_consistency(probe_run_dir, app_config.model.subword_pooling)

        # 1. Load the trained probe bank.
        probe_bank_dir = probe_run_dir / "artifacts" / "probe_bank"
        bank = LayerProbeBank.load(probe_bank_dir)
        logger.info(
            "Loaded probe bank: %d layers, rank=%d, hidden_size=%d, normalization=%s",
            len(bank.probes),
            bank.rank,
            bank.hidden_size,
            bank.input_normalization,
        )

        # 2. Generate stimuli (or load if a cached stimuli artifact exists later).
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

        # 3. Cache or extract activations for all stimuli.
        cache = ActivationCache(
            make_cache_key(
                corpus_name=cfg.activation_cache_name,
                model_name=app_config.model.model_name,
                cache_root=context.cache_dir,
            )
        )
        stimulus_activations = _ensure_stimuli_cache(
            cache=cache,
            stimulus_records=stimulus_set.records,
            app_config=app_config,
        )

        # 4. For each stimulus and each layer, apply the probe and record per-pair distances.
        per_stimulus_predictions: list[dict[str, Any]] = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for probe in bank.probes.values():
            probe.to(device)

        for record in stimulus_set.records:
            if record.stimulus_id not in stimulus_activations:
                logger.warning("No activation for stimulus %s, skipping", record.stimulus_id)
                continue
            sentence_acts = stimulus_activations[record.stimulus_id]
            word_pairs = _word_pairs_for_record(cfg.stimuli_kind, record)
            # sentence_acts shape: (num_layers, num_words, hidden_size)
            for layer_index, probe in bank.probes.items():
                if layer_index >= sentence_acts.shape[0]:
                    continue
                layer_vectors = sentence_acts[layer_index]  # (num_words, hidden_size)
                pair_results = _apply_probe_to_pairs(
                    probe=probe,
                    layer_vectors=layer_vectors,
                    record=record,
                    word_pairs=word_pairs,
                    device=device,
                )
                for pair_label, distance in pair_results.items():
                    per_stimulus_predictions.append(
                        {
                            "stimulus_id": record.stimulus_id,
                            "item_id": record.item_id,
                            "condition": record.condition,
                            "condition_rank": record.metadata.get("condition_rank"),
                            "subexperiment": record.metadata.get("subexperiment"),
                            "layer_index": layer_index,
                            "pair_label": pair_label,
                            "predicted_distance": distance,
                        }
                    )

        per_stim_path = context.artifact_dir / "per_stimulus_predictions.jsonl"
        write_jsonl(per_stim_path, iter(per_stimulus_predictions))

        # 5. Aggregate to per-(layer, pair, condition) means.
        layer_profile = _aggregate_layer_profile(
            per_stimulus_predictions, num_layers=max(bank.probes) + 1
        )
        layer_profile_path = context.artifact_dir / "layer_profile.json"
        write_json(layer_profile_path, layer_profile)

        manifest_path = write_manifest(
            context,
            {
                "probe_run_dir": str(probe_run_dir),
                "input_normalization": bank.input_normalization,
                "subword_pooling": app_config.model.subword_pooling,
                "stimulus_count": len(stimulus_set.records),
                "num_predictions": len(per_stimulus_predictions),
                "artifacts": {
                    "per_stimulus_predictions": str(per_stim_path),
                    "layer_profile": str(layer_profile_path),
                },
            },
        )

        return ExperimentResult(
            experiment_kind=self.kind,
            run_id=context.run_id,
            summary={
                "stimulus_count": len(stimulus_set.records),
                "num_predictions": len(per_stimulus_predictions),
            },
            artifacts={
                "per_stimulus_predictions": per_stim_path,
                "layer_profile": layer_profile_path,
                "manifest": manifest_path,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_stimuli_cache(
    *,
    cache: ActivationCache,
    stimulus_records: list[StimulusRecord],
    app_config: AppConfig,
) -> dict[str, NDArray[np.float32]]:
    """Cache or extract activations for the stimuli, returning a dict by stimulus_id.

    The dict maps stimulus_id to a (num_layers, num_words, hidden_size) array.
    For the small stimulus sets we use, this fits comfortably in memory; if it
    didn't, we'd switch to a per-stimulus on-disk format.
    """
    split = "stimuli"

    # If cached, read all layers and reconstruct per-sentence slices.
    # But only use the cache if it actually covers all the requested stimuli.
    # If the cache was built with a smaller stimulus set (e.g., num_items=12
    # before we scaled up to num_items=1000), we must re-extract so that the
    # new stimuli get their activations.
    requested_ids = {record.stimulus_id for record in stimulus_records}

    if cache.has_metadata(split):
        meta = cache.read_metadata(split)
        sentence_ids = list(meta["sentence_ids"])  # type: ignore[arg-type]
        cached_ids = set(sentence_ids)

        if not requested_ids.issubset(cached_ids):
            missing = requested_ids - cached_ids
            logger.info(
                "Stimulus activation cache is stale (%d/%d requested stimuli missing). "
                "Discarding cache and re-extracting all %d stimuli.",
                len(missing),
                len(requested_ids),
                len(stimulus_records),
            )
            cache.invalidate(split)  # fall through to extraction below
        else:
            sentence_lengths = list(meta["sentence_lengths"])  # type: ignore[arg-type]
            num_layers = int(meta["num_layers"])  # type: ignore[arg-type]
            hidden_size = int(meta["hidden_size"])  # type: ignore[arg-type]
            all_layers: list[NDArray[np.float32]] = [
                cache.read_layer(split=split, layer_index=layer_idx)
                for layer_idx in range(num_layers)
            ]
            stacked = np.stack(all_layers, axis=0)
            starts = np.concatenate([[0], np.cumsum(sentence_lengths)])
            out: dict[str, NDArray[np.float32]] = {}
            for index, sid in enumerate(sentence_ids):
                out[sid] = stacked[:, starts[index] : starts[index + 1], :]
            logger.info(
                "Loaded cached stimulus activations: %d sentences, %d layers",
                len(sentence_ids),
                num_layers,
            )
            return out

    # Extract.
    logger.info("Extracting activations for %d stimuli", len(stimulus_records))
    extractor = LLMActivationExtractor.from_pretrained(app_config.model)
    items = [(record.stimulus_id, record.tokens) for record in stimulus_records]
    extraction = extractor.extract(items)
    if not extraction.items:
        raise RuntimeError("No stimuli successfully extracted")

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
            "subword_pooling": app_config.model.subword_pooling,
            "skipped_sentence_ids": extraction.skipped_sentence_ids,
        },
    )

    out: dict[str, NDArray[np.float32]] = {}
    for layer_index in range(num_layers):
        per_layer = [item.activations[layer_index] for item in extraction.items]
        concatenated = np.concatenate(per_layer, axis=0).astype(np.float32)
        cache.write_layer(split=split, layer_index=layer_index, activations=concatenated)

    # Build the in-memory dict.
    for item in extraction.items:
        out[item.sentence_id] = item.activations.astype(np.float32)

    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _apply_probe_to_pairs(
    *,
    probe: StructuralProbe,
    layer_vectors: NDArray[np.float32],
    record: StimulusRecord,
    word_pairs: Sequence[tuple[str, str, str]],
    device: torch.device,
) -> dict[str, float]:
    """Apply a probe to one sentence's vectors at one layer; return per-pair distances."""
    n_words = layer_vectors.shape[0]
    out: dict[str, float] = {}

    # Materialize the (1, n_words, hidden_size) tensor and run the probe forward
    # once to get the full pairwise distance matrix at this layer.
    vectors = torch.from_numpy(layer_vectors).unsqueeze(0).to(device)
    with torch.inference_mode():
        # Squared predicted distances, shape (1, n, n).
        pred = probe(vectors)
    pred_matrix = pred.squeeze(0).cpu().numpy()

    for left_role, right_role, label in word_pairs:
        if left_role not in record.role_indices or right_role not in record.role_indices:
            continue
        left_idx = record.role_indices[left_role]
        right_idx = record.role_indices[right_role]
        if left_idx >= n_words or right_idx >= n_words:
            logger.debug(
                "Out-of-range role index for stimulus %s (left=%d, right=%d, n=%d)",
                record.stimulus_id,
                left_idx,
                right_idx,
                n_words,
            )
            continue
        out[label] = float(pred_matrix[left_idx, right_idx])

    return out


def _aggregate_layer_profile(
    rows: list[dict[str, Any]],
    *,
    num_layers: int,
) -> dict[str, Any]:
    """Aggregate per-stimulus predictions to per-(layer, pair, condition) means.

    Output structure suitable for layer-profile plotting:
        {
            "layers": [
                {
                    "layer_index": 0,
                    "by_pair_and_condition": {
                        "wh-evb": {"bare": {"mean": x, "n": k}, "infinitival": ..., ...},
                        "wh-esubj": {...},
                        ...
                    }
                },
                ...
            ]
        }
    """
    # Bucket by (layer, pair, condition).
    buckets: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["layer_index"], row["pair_label"], row["condition"])
        buckets[key].append(row["predicted_distance"])

    # Determine pair labels and conditions present.
    pair_labels = sorted({k[1] for k in buckets})
    conditions = sorted({k[2] for k in buckets})

    layers_out: list[dict[str, Any]] = []
    for layer_index in range(num_layers):
        per_pair: dict[str, dict[str, dict[str, float]]] = {}
        any_present = False
        for pair_label in pair_labels:
            per_condition: dict[str, dict[str, float]] = {}
            for condition in conditions:
                values = buckets.get((layer_index, pair_label, condition), [])
                if values:
                    any_present = True
                    arr = np.asarray(values, dtype=np.float64)
                    per_condition[condition] = {
                        "mean": float(arr.mean()),
                        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                        "n": int(len(arr)),
                    }
            if per_condition:
                per_pair[pair_label] = per_condition
        if any_present:
            layers_out.append({"layer_index": layer_index, "by_pair_and_condition": per_pair})

    return {"layers": layers_out}


register_experiment_runner("apply_probes", ApplyProbesRunner)


def _resolve_probe_run_dir(cfg: ApplyProbesConfig, *, output_dir: Path) -> Path:
    """Pick the probe_run_dir, either explicit or auto-resolved from probe_run_name.

    Resolution rules:
        * If ``probe_run_dir`` is set, return it (must exist and contain a probe bank).
        * Else if ``probe_run_name`` is set, scan
          ``output_dir/probe_training/<probe_run_name>-*`` and pick the most-recently
          modified directory that contains a probe_bank artifact. Iterate from
          newest to oldest, skipping empty/incomplete directories.
        * Else raise.

    Raises:
        ValueError: if neither field is set, or if no matching non-empty run dir
            is found.
    """
    if cfg.probe_run_dir is not None and cfg.probe_run_name is not None:
        raise ValueError(
            "Set exactly one of `probe_run_dir` or `probe_run_name`, not both."
        )

    if cfg.probe_run_dir is not None:
        if not _has_probe_bank(cfg.probe_run_dir):
            raise ValueError(
                f"probe_run_dir {cfg.probe_run_dir} does not contain a probe bank "
                f"at artifacts/probe_bank/probe_bank.pt."
            )
        return cfg.probe_run_dir

    if cfg.probe_run_name is None:
        raise ValueError(
            "Either `probe_run_dir` or `probe_run_name` must be set on apply_probes "
            "config."
        )

    parent = output_dir / "probe_training"
    if not parent.is_dir():
        raise ValueError(
            f"No probe-training output directory at {parent}. Train a probe bank "
            f"first, or set `probe_run_dir` explicitly."
        )

    prefix = f"{cfg.probe_run_name}-"

    def _run_id_timestamp(p: Path) -> str:
        """Extract the sortable timestamp from a run directory name.

        Run directory names have the format ``{name}-{config_hash}-{date}-{time}``,
        e.g. ``qwen25_1_5b_ud_ewt-68c84250-20260503-025243``.  The trailing
        ``{date}-{time}`` is a stable lexicographically-sortable timestamp.
        Falls back to ``"0"`` if the name doesn't match the expected format, so
        malformed directories sort last.
        """
        parts = p.name[len(prefix):].split("-")
        # parts = [config_hash, date, time] when there are 3 trailing segments
        if len(parts) >= 3:
            return f"{parts[-2]}-{parts[-1]}"
        # Fallback: use filesystem mtime string so we still get a valid sort key
        return str(p.stat().st_mtime)

    candidates = sorted(
        (p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)),
        key=_run_id_timestamp,
        reverse=True,
    )
    if not candidates:
        raise ValueError(
            f"No probe-training run directories matching {prefix!r} in {parent}."
        )

    skipped: list[Path] = []
    for candidate in candidates:
        if _has_probe_bank(candidate):
            if skipped:
                logger.info(
                    "Auto-resolved probe_run_name %r to %s "
                    "(skipped %d empty/incomplete: %s)",
                    cfg.probe_run_name,
                    candidate.name,
                    len(skipped),
                    [p.name for p in skipped],
                )
            else:
                logger.info(
                    "Auto-resolved probe_run_name %r to %s",
                    cfg.probe_run_name,
                    candidate.name,
                )
            return candidate
        skipped.append(candidate)

    raise ValueError(
        f"Found {len(candidates)} run directories matching {prefix!r} in {parent}, "
        f"but none contained a probe bank at artifacts/probe_bank/probe_bank.pt: "
        f"{[p.name for p in candidates]}"
    )


def _has_probe_bank(run_dir: Path) -> bool:
    return (run_dir / "artifacts" / "probe_bank" / "probe_bank.pt").is_file()


def _check_pooling_consistency(probe_run_dir: Path, current_pooling: str) -> None:
    """Warn or fail if the probe bank was trained with different subword pooling."""
    provenance_path = probe_run_dir / "artifacts" / "training_provenance.json"
    if not provenance_path.is_file():
        # Older runs (pre-v3) don't record this; we can't check, so skip silently.
        return
    import json

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    trained_pooling = provenance.get("subword_pooling")
    if trained_pooling is None:
        return
    if trained_pooling != current_pooling:
        raise ValueError(
            f"Probe bank was trained with subword_pooling={trained_pooling!r} but "
            f"current model config uses {current_pooling!r}. They must match. "
            f"Either change model.subword_pooling to {trained_pooling!r} in the "
            f"apply-probes config, or train a new probe bank with the current "
            f"pooling setting."
        )


__all__ = ["ApplyProbesRunner"]
