"""Source-target stimulus pair generation for activation patching.

Activation-patching trials require identifying *pairs* of stimuli that
differ only in a structural property of interest, and specifying which
position(s) in the residual stream to intervene on. This module
generates those pairs from the existing observational stimulus sets
without requiring new stimulus generation.

For each Tier-1 experiment, this module exposes a function that takes
a stimulus set (or two, for cc) and returns a list of
:class:`TrialSpec` objects. The activation-patching experiment runner
consumes those specs.

Stimuli themselves are not modified; we only build references to
them. This means a single :class:`StimulusSet` can support multiple
patching experiments without duplication.

Why a separate module rather than methods on the stimulus generators:
the pair-generation logic depends on the *patching protocol* (W2/W4/N1)
rather than on the linguistic experiment (wh-extraction or c-command).
Keeping it separate means the patching-specific decisions (which
role-positions to intervene at, source-target pairings) are visible in
one place rather than scattered across stimulus generators.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from .schema import StimulusRecord, StimulusSet

# ---------------------------------------------------------------------------
# Trial specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """One activation-patching trial.

    A trial is the smallest unit the experiment runner can dispatch:
    one source forward pass to cache residuals at a particular cell, one
    patched target forward pass that splices the source's value in at
    the intervention site, and one probe-application to read out the
    patched target's structural representation.

    All sentence-level identification is via stimulus_id. Resolving a
    stimulus_id to a :class:`StimulusRecord` is the runner's
    responsibility.
    """

    trial_id: str
    """Globally-unique identifier for this trial."""

    experiment_kind: Literal["w2", "w4", "n1"]
    """Which Tier-1 experiment this trial belongs to."""

    item_id: str
    """Underlying lexical item shared by source and target (for clustered stats)."""

    source_stimulus_id: str
    target_stimulus_id: str

    intervention_role: str
    """Name of the role at which to intervene (e.g., ``"embedded_subject"``).
    The runner resolves this to subword indices via word-to-subword
    alignment for both source and target."""

    intervention_layer: int
    """Index into ``hidden_states`` (i.e., the residual stream
    *immediately before* this transformer block runs)."""

    measurement_layers: tuple[int, ...]
    """Layers at which to apply the probe to the patched target run.
    For W2 typically a single layer (the observational peak). For W4
    can be a sweep."""

    measurement_pairs: tuple[tuple[str, str, str], ...]
    """List of ``(role_left, role_right, pair_label)`` tuples to compute
    probe distances for. For wh experiments typically just
    ``("wh", "embedded_subject", "wh-esubj")``."""

    metadata: dict = field(default_factory=dict)
    """Free-form: source/target conditions, modifier_type for cc, etc.
    Recorded to the per-trial JSONL for downstream stats."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def filter_lexical_dp_subject_items(
    stimulus_set: StimulusSet,
) -> list[StimulusRecord]:
    """Return only the records belonging to items whose embedded subject
    is a *lexical DP* (i.e., lexically identical across all conditions
    of the item, with no overt case morphology).

    Used by W2 and W4 to eliminate the case-marking confound at the
    ``embedded_subject`` position. The check is structural rather than
    string-matching: an item is kept iff all of its records share the
    same ``embedded_subject_surface`` metadata. This works equally for
    "the student" / "the student" / "the student" and for any future
    full-DP items added to the lexicon.
    """
    by_item = stimulus_set.by_item()
    keep_item_ids: set[str] = set()
    for item_id, records in by_item.items():
        surfaces = {
            r.metadata.get("embedded_subject_surface", "")
            for r in records
        }
        if len(surfaces) == 1 and next(iter(surfaces)):
            keep_item_ids.add(item_id)
    return [r for r in stimulus_set.records if r.item_id in keep_item_ids]


# ---------------------------------------------------------------------------
# Tier-1 trial generators
# ---------------------------------------------------------------------------


# Standard wh measurement pair list. We measure all three pairs at every
# trial for completeness; downstream stats subset to ``wh-esubj`` for the
# headline finding but the full set is useful for diagnostic checks.
_WH_MEASUREMENT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("wh", "embedded_subject", "wh-esubj"),
    ("wh", "embedded_verb", "wh-evb"),
    ("embedded_subject", "embedded_verb", "esubj-evb"),
)


def generate_w2_trials(
    *,
    stimulus_set: StimulusSet,
    intervention_layer_per_model: int,
    intervention_roles: Sequence[str] = ("wh", "embedded_subject", "embedded_verb"),
    source_target_pairs: Sequence[tuple[str, str]] = (("finite", "bare"), ("infinitival", "bare")),
    filter_to_lexical_dp: bool = True,
) -> list[TrialSpec]:
    """Generate W2 (phase-edge specificity) trials.

    Output cardinality:
        len(items) × len(intervention_roles) × len(source_target_pairs)

    With the default settings on the standard wh stimulus set
    (~430 lexical-DP items, 3 roles, 2 pairs) this is ~2580 trials per
    model. Each trial requires one source forward pass (cached) and one
    patched target forward pass. Compute is dominated by the target
    passes, batched.

    Args:
        stimulus_set: full wh-extraction stimulus set with all 3
            conditions per item.
        intervention_layer_per_model: index of the layer to patch at,
            typically the model's β-peak observational layer for the
            finite-bare contrast on wh-esubj. The same layer is used as
            both ``L_patch`` and the single ``L_measure`` in W2.
        intervention_roles: which roles to intervene at.
        source_target_pairs: list of ``(source_condition, target_condition)``.
        filter_to_lexical_dp: whether to filter to items where the
            embedded subject is lexically identical across conditions
            (eliminating the case-marking confound).
    """
    if filter_to_lexical_dp:
        records = filter_lexical_dp_subject_items(stimulus_set)
    else:
        records = list(stimulus_set.records)

    # Index by (item_id, condition) for pair construction.
    by_item_condition: dict[tuple[str, str], StimulusRecord] = {}
    for r in records:
        by_item_condition[(r.item_id, r.condition)] = r

    trials: list[TrialSpec] = []
    item_ids = sorted({r.item_id for r in records})
    for item_id in item_ids:
        for source_cond, target_cond in source_target_pairs:
            source = by_item_condition.get((item_id, source_cond))
            target = by_item_condition.get((item_id, target_cond))
            if source is None or target is None:
                continue
            for role in intervention_roles:
                # Skip embedded_verb when source-target tense morphology
                # differs (finite-vs-bare or finite-vs-infinitival): the
                # token at this position is "ate" vs "eat", a lexical
                # confound. Embedded_verb is only patched for the
                # (infinitival, bare) pair, where both share "eat".
                if role == "embedded_verb" and "finite" in (source_cond, target_cond):
                    continue
                trial_id = (
                    f"w2|{item_id}|{source_cond}->{target_cond}|{role}"
                    f"|L{intervention_layer_per_model}"
                )
                trials.append(
                    TrialSpec(
                        trial_id=trial_id,
                        experiment_kind="w2",
                        item_id=item_id,
                        source_stimulus_id=source.stimulus_id,
                        target_stimulus_id=target.stimulus_id,
                        intervention_role=role,
                        intervention_layer=intervention_layer_per_model,
                        measurement_layers=(intervention_layer_per_model,),
                        measurement_pairs=_WH_MEASUREMENT_PAIRS,
                        metadata={
                            "source_condition": source_cond,
                            "target_condition": target_cond,
                            "lexical_dp_filtered": filter_to_lexical_dp,
                        },
                    )
                )
    return trials


def generate_w4_trials(
    *,
    stimulus_set: StimulusSet,
    intervention_layers: Sequence[int],
    measurement_layer: int,
    intervention_role: str = "embedded_subject",
    source_target_pair: tuple[str, str] = ("finite", "bare"),
    filter_to_lexical_dp: bool = True,
) -> list[TrialSpec]:
    """Generate W4 (layer-localization) trials.

    For each item, sweeps over ``intervention_layers`` (the L_patch
    sweep). All trials use a single ``intervention_role`` (default:
    ``embedded_subject``) and a single source-target pair (default:
    finite-bare). Measurement is at one ``measurement_layer`` (typically
    the model's observational β-peak), but the runner can be configured
    to measure at multiple layers per trial as a stage-2 extension.

    Output cardinality:
        len(items) × len(intervention_layers).
    """
    if filter_to_lexical_dp:
        records = filter_lexical_dp_subject_items(stimulus_set)
    else:
        records = list(stimulus_set.records)

    by_item_condition: dict[tuple[str, str], StimulusRecord] = {}
    for r in records:
        by_item_condition[(r.item_id, r.condition)] = r

    source_cond, target_cond = source_target_pair
    item_ids = sorted({r.item_id for r in records})
    trials: list[TrialSpec] = []
    for item_id in item_ids:
        source = by_item_condition.get((item_id, source_cond))
        target = by_item_condition.get((item_id, target_cond))
        if source is None or target is None:
            continue
        for L_patch in intervention_layers:
            trial_id = (
                f"w4|{item_id}|{source_cond}->{target_cond}|{intervention_role}"
                f"|patch{L_patch}|measure{measurement_layer}"
            )
            trials.append(
                TrialSpec(
                    trial_id=trial_id,
                    experiment_kind="w4",
                    item_id=item_id,
                    source_stimulus_id=source.stimulus_id,
                    target_stimulus_id=target.stimulus_id,
                    intervention_role=intervention_role,
                    intervention_layer=L_patch,
                    measurement_layers=(measurement_layer,),
                    measurement_pairs=_WH_MEASUREMENT_PAIRS,
                    metadata={
                        "source_condition": source_cond,
                        "target_condition": target_cond,
                        "lexical_dp_filtered": filter_to_lexical_dp,
                    },
                )
            )
    return trials


# Standard cc reflexive measurement pairs. The roles ``subject``,
# ``anaphor``, and ``modifier`` are populated by the cc reflexive
# stimulus generator; ``modifier`` points at the modifier-internal NP
# (e.g., "John" in "the friend of John"), which is the structurally-
# closer-but-not-c-commanding NP we measure against the anaphor.
_CC_REFLEXIVE_MEASUREMENT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("anaphor", "subject", "anaphor-subject"),
    ("anaphor", "modifier", "anaphor-modifier"),
)


def generate_n1_trials(
    *,
    stimulus_set: StimulusSet,
    intervention_layer_per_model: int,
    bidirectional: bool = True,
) -> list[TrialSpec]:
    """Generate N1 (per-layer robustness causality) trials.

    For each cc reflexive item × modifier-type, build trials for the
    match → swap and (if bidirectional) swap → match patches. The
    intervention position is the anaphor token, which is lexically
    identical between match and swap configurations by construction.

    Output cardinality:
        len(items) × 4 modifier types × (2 if bidirectional else 1).
    """
    # Index by (item_id, modifier_type, gender_config) over reflexive
    # records only. The cc reflexive stimulus generator sets
    # ``anaphor_type``, ``modifier_type``, and ``gender_config`` ("match"
    # or "swap") in metadata. We restrict to ``anaphor_type ==
    # "reflexive"`` because N1's intervention is at the anaphor token
    # and we want the himself/herself form, not the pronoun him/her
    # form (which is in the same stimulus generator but tests
    # Principle B differently).
    by_key: dict[tuple[str, str, str], StimulusRecord] = {}
    for r in stimulus_set.records:
        meta = r.metadata
        if meta.get("subexperiment") != "reflexive":
            continue
        if meta.get("anaphor_type") != "reflexive":
            continue
        modifier_type = meta.get("modifier_type")
        gender_config = meta.get("gender_config")
        if modifier_type is None or gender_config not in ("match", "swap"):
            continue
        by_key[(r.item_id, str(modifier_type), str(gender_config))] = r

    trials: list[TrialSpec] = []
    item_ids = sorted({key[0] for key in by_key})
    modifier_types = ("pp", "poss", "rcobj", "rcsubj")
    direction_pairs: tuple[tuple[str, str], ...] = (
        (("match", "swap"), ("swap", "match")) if bidirectional
        else (("match", "swap"),)
    )

    for item_id in item_ids:
        for mod in modifier_types:
            for src_cfg, tgt_cfg in direction_pairs:
                source = by_key.get((item_id, mod, src_cfg))
                target = by_key.get((item_id, mod, tgt_cfg))
                if source is None or target is None:
                    continue
                trial_id = (
                    f"n1|{item_id}|{mod}|{src_cfg}->{tgt_cfg}"
                    f"|L{intervention_layer_per_model}"
                )
                trials.append(
                    TrialSpec(
                        trial_id=trial_id,
                        experiment_kind="n1",
                        item_id=item_id,
                        source_stimulus_id=source.stimulus_id,
                        target_stimulus_id=target.stimulus_id,
                        intervention_role="anaphor",
                        intervention_layer=intervention_layer_per_model,
                        measurement_layers=(intervention_layer_per_model,),
                        measurement_pairs=_CC_REFLEXIVE_MEASUREMENT_PAIRS,
                        metadata={
                            "modifier_type": mod,
                            "source_gender_config": src_cfg,
                            "target_gender_config": tgt_cfg,
                        },
                    )
                )
    return trials


__all__ = [
    "TrialSpec",
    "filter_lexical_dp_subject_items",
    "generate_n1_trials",
    "generate_w2_trials",
    "generate_w4_trials",
]
