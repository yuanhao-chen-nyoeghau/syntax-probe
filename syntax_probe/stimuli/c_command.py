"""C-command stimulus generators.

Three sub-experiments probing whether LLMs encode c-command relations beyond
what surface UD parses encode:

    reflexive  : reflexive vs. pronoun binding by a c-commanding subject vs.
                 a non-c-commanding modifier-internal NP (Binding Theory
                 Principles A and B). Three modifier types:
                   pp        - "The senator near the lobbyist praised himself."
                   possessor - "The senator's lobbyist praised himself."
                               (modifier-internal NP precedes the c-commanding
                                head; this is NOT a linear-vs-c-command test
                                because both linear distance and c-command pick
                                the head N2. It does defeat a
                                first-NP-in-subject heuristic, which would
                                pick the possessor N1. See the methodological
                                note at the bottom of this docstring.)
                   rc_subj   - "The senator who befriended the lobbyist praised himself."
                   rc_obj    - "The senator who the lobbyist befriended praised himself."
                 In all variants, the matrix subject c-commands the anaphor;
                 the modifier-internal NP does not.

    principle_c: Cataphora preference (Binding Principle C). R-expression
                 may not be c-commanded by a coreferent pronoun.
                   violation - "He claimed the senator left."
                               (he c-commands the senator; coreference forbidden)
                   obviated  - "When he was elected, the senator left."
                               (he is in an adjunct; doesn't c-command;
                                coreference permitted)

    bound_var  : Bound-variable anaphora. Quantifier (every/each) must
                 c-command a pronoun for a bound-variable reading.
                   subj_quant - "Every senator near the lobbyists praised his cat."
                                (every senator c-commands his -> bound reading available)
                   mod_quant  - "The lobbyist near every senator praised his cat."
                                (every senator does NOT c-command his ->
                                 bound reading not available at SS;
                                 may or may not be available at LF if LM does QR)

The same probe-bank (trained on UD) is applied to all three. Each sub-experiment
defines its own role-pair labels (see PAIRS_BY_SUBEXPERIMENT in apply_probes).

==============================================================================
Methodological note: what the four modifier types actually defeat
==============================================================================

The four modifier types are sometimes described as testing "c-command vs.
linear distance" uniformly. That is misleading. The four configurations defeat
DIFFERENT baseline heuristics, and only three of them are linear-vs-structural
tests:

  template                                        c-commander          modifier-internal NP
  --------------------------------------------    -----------          ------------------------
  pp:    "Det N1 P Det N2 V REFL"                 N1 (subject head)    N2 (inside PP modifier)
  rc:    "Det N1 [who ... N2 ...] V REFL"         N1 (subject head)    N2 (inside RC)
  poss:  "Det N1's N2 V REFL"                     N2 (subject head)    N1 (possessor)

Linear distance from the anaphor to each NP:

  pp:    head N1 is farther; modifier-internal N2 is closer    -> linear predicts WRONG
  rc:    head N1 is farther; modifier-internal N2 is closer    -> linear predicts WRONG
  poss:  head N2 is closer;  modifier-internal N1 is farther   -> linear predicts RIGHT

UD tree distance from the anaphor to each NP (the probe's training target):
in all four configurations the subject head is closer than the modifier-internal
NP, since the head is in the matrix subject slot (sister to the matrix VP) and
the modifier-internal NP is further embedded inside the subject DP. UD distance
always agrees with c-command on these configurations; this is a property of the
probe's training target, not of c-command per se.

The empirical content of each configuration:

  pp, rc_subj, rc_obj:  test c-command against a LINEAR-PROXIMITY heuristic.
                        A model that picks the linearly-closest NP as the
                        antecedent gets these wrong (it picks the
                        modifier-internal NP); c-command and UD distance both
                        predict the head.

  poss:                 NOT a linear-vs-c-command test. Linear and c-command
                        agree on the head. The possessor case defeats a
                        DIFFERENT baseline: "first-NP-in-subject" / "leftmost
                        NP in the subject DP is the antecedent". A model
                        following that heuristic picks the possessor; c-command
                        and UD distance both predict the head.

A model that passes all four configurations can't be relying on either
linear proximity OR first-NP-in-subject. The four configurations together
provide stronger refutation of baseline heuristics than any one of them alone.
This is the strong sense in which the cross-modifier comparison is informative;
it is not, however, evidence that the LLM specifically encodes c-command rather
than UD distance, since the probe is trained on UD and UD predicts the same
ordering in all four cases.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..utils.text import tokenize_words
from .schema import StimulusRecord, StimulusSet

# ---------------------------------------------------------------------------
# Condition rank table — used for ordering plots and tables consistently.
# ---------------------------------------------------------------------------

_CONDITION_RANK_BASE: dict[str, int] = {
    # Reflexive binding — modifier_type x anaphor_type x gender_config
    "refl_refl_match_pp": 11,
    "refl_refl_swap_pp": 12,
    "refl_pron_match_pp": 13,
    "refl_pron_swap_pp": 14,
    "refl_refl_match_poss": 21,
    "refl_refl_swap_poss": 22,
    "refl_pron_match_poss": 23,
    "refl_pron_swap_poss": 24,
    "refl_refl_match_rcsubj": 31,
    "refl_refl_swap_rcsubj": 32,
    "refl_pron_match_rcsubj": 33,
    "refl_pron_swap_rcsubj": 34,
    "refl_refl_match_rcobj": 41,
    "refl_refl_swap_rcobj": 42,
    "refl_pron_match_rcobj": 43,
    "refl_pron_swap_rcobj": 44,
    # Principle C
    "prinC_violation": 51,
    "prinC_obviated": 52,
    # Bound variable
    "bv_subj_quant": 61,
    "bv_mod_quant": 62,
}


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CCommandLexicon:
    """Lexical inventory shared across the three sub-experiments."""

    masculine_nouns: tuple[str, ...]
    """Singular masculine-leaning role nouns; full DPs use 'the X' or 'X's Y'."""

    feminine_nouns: tuple[str, ...]
    """Singular feminine-leaning role nouns."""

    plural_nouns: tuple[str, ...]
    """Plural fillers for bound-variable PP-internal NPs."""

    prepositions: tuple[str, ...]
    """Prepositions licensing PP-modifiers of NPs."""

    matrix_verbs_past: tuple[str, ...]
    """Past-tense transitive verbs licensing animate objects (used for
    reflexive-binding sub-experiment with reflexive/pronoun direct object,
    and for the matrix verb in bound-variable stimuli)."""

    rc_verbs_past: tuple[str, ...]
    """Past-tense transitive verbs used inside relative clauses. Disjoint
    from `matrix_verbs_past` so a single item never repeats a verb."""

    cp_verbs_past: tuple[str, ...]
    """Past-tense verbs taking a finite that-CP complement; used for
    Principle C violation cases ('He claimed the senator left')."""

    pred_verbs_past: tuple[str, ...]
    """Intransitive past-tense predicates used as embedded VP in Principle C
    and as adjunct/main predicates in obviated cases."""

    poss_objects: tuple[str, ...]
    """Singular possessable nouns ('cat', 'manager') used for 'his/her X'
    in bound-variable stimuli."""


DEFAULT_LEXICON: CCommandLexicon = CCommandLexicon(
    masculine_nouns=(
        "actor", "boy", "father", "groom", "husband",
        "king", "monk", "prince", "salesman", "uncle",
    ),
    feminine_nouns=(
        "actress", "girl", "mother", "bride", "wife",
        "queen", "nun", "princess", "saleswoman", "aunt",
    ),
    plural_nouns=(
        "lawyers", "officers", "judges", "doctors", "students",
    ),
    prepositions=(
        "near", "behind", "beside", "with", "against", "by",
    ),
    matrix_verbs_past=(
        "praised", "criticized", "blamed", "defended", "described",
        "studied", "watched", "questioned", "examined", "introduced",
    ),
    rc_verbs_past=(
        "befriended", "respected", "trusted", "remembered",
        "imitated", "thanked", "scolded", "interviewed",
    ),
    cp_verbs_past=(
        "claimed", "insisted", "announced", "reported",
        "argued", "declared",
    ),
    pred_verbs_past=(
        "left", "won", "resigned", "celebrated",
        "succeeded", "complained",
    ),
    poss_objects=(
        "manager", "agent", "lawyer", "doctor", "secretary",
    ),
)


# ---------------------------------------------------------------------------
# Anaphor / pronoun lookup
# ---------------------------------------------------------------------------

_REFLEXIVE_BY_GENDER: dict[str, str] = {"masculine": "himself", "feminine": "herself"}
_PRONOUN_BY_GENDER:   dict[str, str] = {"masculine": "him", "feminine": "her"}
_SUBJECT_PRONOUN_BY_GENDER: dict[str, str] = {"masculine": "he", "feminine": "she"}
_POSS_PRONOUN_BY_GENDER:    dict[str, str] = {"masculine": "his", "feminine": "her"}


# ===========================================================================
# Sub-experiment 1: Reflexive binding
# ===========================================================================


@dataclass(frozen=True, slots=True)
class ReflexiveItem:
    """A lexical item for reflexive-binding stimuli.

    Realized in 16 conditions per item:
      4 modifier types (pp, poss, rcsubj, rcobj)
      x 2 anaphor types (refl, pron)
      x 2 gender configs (match, swap)

    `n_match` is the noun whose gender aligns with the anaphor in the
    'match' condition (it then sits in the subject slot, the c-commander).
    `n_other` is the opposite-gender noun (modifier-internal in 'match').
    In 'swap', the roles flip: n_other is the subject (gender-mismatched
    c-commander), n_match is modifier-internal (gender-matched non-c-commander).
    """

    item_id: str
    n_match: str
    n_other: str
    preposition: str
    matrix_verb_past: str
    rc_verb_past: str
    """Distinct from matrix_verb_past."""
    gender: str  # 'masculine' | 'feminine'


_REFL_MODIFIER_TYPES = ("pp", "poss", "rcsubj", "rcobj")


def _enumerate_reflexive_items(
    lexicon: CCommandLexicon, *, seed: int, shuffle: bool,
) -> list[ReflexiveItem]:
    items: list[ReflexiveItem] = []
    index = 0
    for gender in ("masculine", "feminine"):
        same = lexicon.masculine_nouns if gender == "masculine" else lexicon.feminine_nouns
        opp  = lexicon.feminine_nouns  if gender == "masculine" else lexicon.masculine_nouns
        for n_match in same:
            for n_other in opp:
                for prep in lexicon.prepositions:
                    for verb in lexicon.matrix_verbs_past:
                        for rc_verb in lexicon.rc_verbs_past:
                            items.append(ReflexiveItem(
                                item_id=f"refl-{index:07d}",
                                n_match=n_match, n_other=n_other,
                                preposition=prep,
                                matrix_verb_past=verb, rc_verb_past=rc_verb,
                                gender=gender,
                            ))
                            index += 1
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    return items


def _render_reflexive(
    unique_item_id: str,
    item: ReflexiveItem,
    *,
    anaphor_type: str,    # 'refl' or 'pron'
    gender_config: str,   # 'match' or 'swap'
    modifier_type: str,   # 'pp' | 'poss' | 'rcsubj' | 'rcobj'
) -> StimulusRecord:
    """Render one reflexive-binding stimulus for one condition."""
    if anaphor_type == "refl":
        anaphor = _REFLEXIVE_BY_GENDER[item.gender]
    elif anaphor_type == "pron":
        anaphor = _PRONOUN_BY_GENDER[item.gender]
    else:
        raise ValueError(f"unknown anaphor_type {anaphor_type!r}")

    if gender_config == "match":
        subject_noun, modifier_noun = item.n_match, item.n_other
    elif gender_config == "swap":
        subject_noun, modifier_noun = item.n_other, item.n_match
    else:
        raise ValueError(f"unknown gender_config {gender_config!r}")

    verb = item.matrix_verb_past
    rc_verb = item.rc_verb_past

    if modifier_type == "pp":
        text = (
            f"The {subject_noun} {item.preposition} the {modifier_noun} "
            f"{verb} {anaphor}."
        )
    elif modifier_type == "poss":
        # "The X's Y praised himself." Y is the head/subject (c-commander),
        # X is the genitive specifier (modifier-internal, NOT a c-commander).
        # In our naming, subject_noun is the c-commander (head Y), and
        # modifier_noun is the possessor X — which appears LINEARLY FIRST.
        text = (
            f"The {modifier_noun}'s {subject_noun} "
            f"{verb} {anaphor}."
        )
    elif modifier_type == "rcsubj":
        # "The senator who befriended the lobbyist praised himself."
        text = (
            f"The {subject_noun} who {rc_verb} the {modifier_noun} "
            f"{verb} {anaphor}."
        )
    elif modifier_type == "rcobj":
        # "The senator who the lobbyist befriended praised himself."
        text = (
            f"The {subject_noun} who the {modifier_noun} {rc_verb} "
            f"{verb} {anaphor}."
        )
    else:
        raise ValueError(f"unknown modifier_type {modifier_type!r}")

    tokens = tokenize_words(text)

    role_indices: dict[str, int] = {}
    role_indices["subject"] = _require(
        _find_subseq_after(tokens, [subject_noun], 0),
        "subject", text,
    )
    # In poss the modifier appears BEFORE the subject linearly; for the other
    # templates it appears AFTER. Search from the start in poss.
    mod_search_start = 0 if modifier_type == "poss" else role_indices["subject"] + 1
    role_indices["modifier"] = _require(
        _find_subseq_after(tokens, [modifier_noun], mod_search_start),
        "modifier", text,
    )
    role_indices["matrix_verb"] = _require(
        _find_subseq_after(
            tokens, [verb],
            max(role_indices["subject"], role_indices["modifier"]) + 1,
        ),
        "matrix_verb", text,
    )
    role_indices["anaphor"] = _require(
        _find_subseq_after(tokens, [anaphor], role_indices["matrix_verb"] + 1),
        "anaphor", text,
    )

    condition = f"refl_{anaphor_type}_{gender_config}_{modifier_type}"
    return StimulusRecord(
        stimulus_id=f"{unique_item_id}-{condition}",
        item_id=unique_item_id,
        condition=condition,
        text=" ".join(tokens),
        tokens=tokens,
        role_indices=role_indices,
        metadata={
            "subexperiment": "reflexive",
            "condition_rank": _CONDITION_RANK_BASE[condition],
            "anaphor_type": "reflexive" if anaphor_type == "refl" else "pronoun",
            "gender_config": gender_config,
            "modifier_type": modifier_type,
            "gender": item.gender,
            "subject_surface": subject_noun,
            "modifier_surface": modifier_noun,
            "anaphor_surface": anaphor,
            "preposition": item.preposition,
            "matrix_verb_past": verb,
            "rc_verb_past": rc_verb,
            "lexical_item_id": item.item_id,
            "rendered_text": text,
        },
    )


# ===========================================================================
# Sub-experiment 2: Principle C (cataphora)
# ===========================================================================


@dataclass(frozen=True, slots=True)
class PrincipleCItem:
    """Lexical item for Principle C stimuli.

    Two conditions:
      violation : "He claimed the senator left."
                  (cataphoric pronoun c-commands the R-expression)
      obviated  : "When he was elected, the senator celebrated."
                  (cataphoric pronoun in adjunct; doesn't c-command)
    """

    item_id: str
    r_expression: str
    cp_verb_past: str
    embedded_pred: str
    adjunct_pred: str
    gender: str  # 'masculine' | 'feminine'


_ADJUNCT_PREDS: tuple[str, ...] = (
    "was elected",
    "was indicted",
    "was promoted",
    "was hired",
)


def _enumerate_principlec_items(
    lexicon: CCommandLexicon, *, seed: int, shuffle: bool,
) -> list[PrincipleCItem]:
    items: list[PrincipleCItem] = []
    index = 0
    for gender in ("masculine", "feminine"):
        nouns = lexicon.masculine_nouns if gender == "masculine" else lexicon.feminine_nouns
        for noun in nouns:
            for cp_verb in lexicon.cp_verbs_past:
                for embedded in lexicon.pred_verbs_past:
                    for adj in _ADJUNCT_PREDS:
                        items.append(PrincipleCItem(
                            item_id=f"prinC-{index:06d}",
                            r_expression=noun,
                            cp_verb_past=cp_verb,
                            embedded_pred=embedded,
                            adjunct_pred=adj,
                            gender=gender,
                        ))
                        index += 1
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    return items


def _render_principlec(
    unique_item_id: str,
    item: PrincipleCItem,
    condition: str,  # 'violation' | 'obviated'
) -> StimulusRecord:
    pron = _SUBJECT_PRONOUN_BY_GENDER[item.gender]

    if condition == "violation":
        text = (
            f"{pron.capitalize()} {item.cp_verb_past} the {item.r_expression} "
            f"{item.embedded_pred}."
        )
    elif condition == "obviated":
        text = (
            f"When {pron} {item.adjunct_pred}, the {item.r_expression} "
            f"{item.embedded_pred}."
        )
    else:
        raise ValueError(f"unknown principle_c condition {condition!r}")

    tokens = tokenize_words(text)

    # Find the pronoun (case-insensitive: "He" sentence-initial in violation,
    # "he" sentence-medial in obviated).
    pron_idx = None
    for i, tok in enumerate(tokens):
        if tok.lower() == pron:
            pron_idx = i
            break
    role_indices: dict[str, int] = {}
    role_indices["pronoun"] = _require(pron_idx, "pronoun", text)
    role_indices["r_expression"] = _require(
        _find_subseq_after(tokens, [item.r_expression], role_indices["pronoun"] + 1),
        "r_expression", text,
    )

    full_condition = f"prinC_{condition}"
    return StimulusRecord(
        stimulus_id=f"{unique_item_id}-{full_condition}",
        item_id=unique_item_id,
        condition=full_condition,
        text=" ".join(tokens),
        tokens=tokens,
        role_indices=role_indices,
        metadata={
            "subexperiment": "principle_c",
            "condition_rank": _CONDITION_RANK_BASE[full_condition],
            "ccommands": condition == "violation",
            "gender": item.gender,
            "r_expression": item.r_expression,
            "pronoun_surface": pron,
            "cp_verb_past": item.cp_verb_past,
            "embedded_pred": item.embedded_pred,
            "adjunct_pred": item.adjunct_pred,
            "lexical_item_id": item.item_id,
            "rendered_text": text,
        },
    )


# ===========================================================================
# Sub-experiment 3: Bound-variable anaphora
# ===========================================================================


@dataclass(frozen=True, slots=True)
class BoundVarItem:
    """Lexical item for bound-variable stimuli.

    Two conditions:
      subj_quant : "Every senator near the lobbyists praised his cat."
                   (every senator c-commands his -> bound reading at SS)
      mod_quant  : "The lobbyist near every senator praised his cat."
                   (every senator does NOT c-command his at SS)
    """

    item_id: str
    quant_noun: str
    """Noun headed by 'every' (e.g. 'senator')."""
    other_noun: str
    """The other noun (e.g. 'lobbyist'); same gender as quant_noun."""
    plural_filler: str
    """Plural filler used in subj_quant's PP ('the lobbyers')."""
    preposition: str
    matrix_verb_past: str
    poss_object: str
    """Singular possessable noun ('cat', 'manager')."""
    gender: str


def _enumerate_boundvar_items(
    lexicon: CCommandLexicon, *, seed: int, shuffle: bool,
) -> list[BoundVarItem]:
    items: list[BoundVarItem] = []
    index = 0
    for gender in ("masculine", "feminine"):
        nouns = lexicon.masculine_nouns if gender == "masculine" else lexicon.feminine_nouns
        for q in nouns:
            for other in nouns:
                if other == q:
                    continue
                for plural in lexicon.plural_nouns:
                    for prep in lexicon.prepositions:
                        for verb in lexicon.matrix_verbs_past:
                            for poss_obj in lexicon.poss_objects:
                                items.append(BoundVarItem(
                                    item_id=f"bv-{index:07d}",
                                    quant_noun=q, other_noun=other,
                                    plural_filler=plural,
                                    preposition=prep,
                                    matrix_verb_past=verb,
                                    poss_object=poss_obj,
                                    gender=gender,
                                ))
                                index += 1
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    return items


def _render_boundvar(
    unique_item_id: str,
    item: BoundVarItem,
    condition: str,  # 'subj_quant' | 'mod_quant'
) -> StimulusRecord:
    poss_pron = _POSS_PRONOUN_BY_GENDER[item.gender]

    if condition == "subj_quant":
        text = (
            f"Every {item.quant_noun} {item.preposition} the {item.plural_filler} "
            f"{item.matrix_verb_past} {poss_pron} {item.poss_object}."
        )
    elif condition == "mod_quant":
        text = (
            f"The {item.other_noun} {item.preposition} every {item.quant_noun} "
            f"{item.matrix_verb_past} {poss_pron} {item.poss_object}."
        )
    else:
        raise ValueError(f"unknown bound_var condition {condition!r}")

    tokens = tokenize_words(text)
    role_indices: dict[str, int] = {}

    if condition == "subj_quant":
        role_indices["quantifier"] = 0  # 'Every'
        role_indices["quant_noun"] = _require(
            _find_subseq_after(tokens, [item.quant_noun], 1),
            "quant_noun", text,
        )
    else:
        every_idx = None
        for i, t in enumerate(tokens):
            if t.lower() == "every":
                every_idx = i
                break
        role_indices["quantifier"] = _require(every_idx, "quantifier", text)
        role_indices["quant_noun"] = _require(
            _find_subseq_after(tokens, [item.quant_noun], role_indices["quantifier"] + 1),
            "quant_noun", text,
        )

    role_indices["matrix_verb"] = _require(
        _find_subseq_after(tokens, [item.matrix_verb_past], role_indices["quant_noun"] + 1),
        "matrix_verb", text,
    )
    role_indices["pronoun"] = _require(
        _find_subseq_after(tokens, [poss_pron], role_indices["matrix_verb"] + 1),
        "pronoun", text,
    )

    full_condition = f"bv_{condition}"
    return StimulusRecord(
        stimulus_id=f"{unique_item_id}-{full_condition}",
        item_id=unique_item_id,
        condition=full_condition,
        text=" ".join(tokens),
        tokens=tokens,
        role_indices=role_indices,
        metadata={
            "subexperiment": "bound_var",
            "condition_rank": _CONDITION_RANK_BASE[full_condition],
            "ccommands_at_ss": condition == "subj_quant",
            "gender": item.gender,
            "quant_noun": item.quant_noun,
            "other_noun": item.other_noun,
            "plural_filler": item.plural_filler,
            "preposition": item.preposition,
            "matrix_verb_past": item.matrix_verb_past,
            "poss_object": item.poss_object,
            "lexical_item_id": item.item_id,
            "rendered_text": text,
        },
    )


# ===========================================================================
# Top-level public API
# ===========================================================================


def generate_c_command_stimuli(
    num_items: int = 1000,
    *,
    subexperiments: tuple[str, ...] = ("reflexive", "principle_c", "bound_var"),
    lexicon: CCommandLexicon = DEFAULT_LEXICON,
    seed: int = 0,
) -> StimulusSet:
    """Generate stimuli for one or more c-command sub-experiments.

    Args:
        num_items: number of base items per included sub-experiment. Each
            item produces:
                - reflexive   : 16 conditions
                - principle_c : 2 conditions
                - bound_var   : 2 conditions
            so total stimuli = num_items * (16 + 2 + 2) = 20 * num_items
            with all three included.
        subexperiments: which sub-experiments to include. Default = all three.
        lexicon: shared lexicon.
        seed: rng seed for shuffling combinatorial pools (offsets used per
            sub-experiment so the same `seed` doesn't shuffle different pools
            identically).
    """
    if num_items < 1:
        raise ValueError("num_items must be at least 1")

    valid = {"reflexive", "principle_c", "bound_var"}
    bad = set(subexperiments) - valid
    if bad:
        raise ValueError(f"unknown subexperiments: {bad}")

    records: list[StimulusRecord] = []

    if "reflexive" in subexperiments:
        records.extend(_generate_reflexive(num_items, lexicon, seed))
    if "principle_c" in subexperiments:
        records.extend(_generate_principlec(num_items, lexicon, seed + 1))
    if "bound_var" in subexperiments:
        records.extend(_generate_boundvar(num_items, lexicon, seed + 2))

    return StimulusSet(
        name="c_command",
        experiment="c_command",
        records=records,
        description=(
            "C-command stimuli: reflexive binding (16 conditions per item: "
            "4 modifier types x reflexive/pronoun x match/swap gender), "
            "Principle C cataphora (violation vs. obviated), and bound-variable "
            "anaphora (subject vs. modifier-internal quantifier)."
        ),
    )


def _generate_reflexive(
    num_items: int, lexicon: CCommandLexicon, seed: int,
) -> list[StimulusRecord]:
    pool = _enumerate_reflexive_items(lexicon, seed=seed, shuffle=True)
    if not pool:
        raise ValueError("Reflexive item pool is empty.")
    out: list[StimulusRecord] = []
    for idx in range(num_items):
        item = pool[idx % len(pool)]
        unique_id = item.item_id if num_items <= len(pool) else f"{item.item_id}-{idx:05d}"
        for modifier_type in _REFL_MODIFIER_TYPES:
            for anaphor_type in ("refl", "pron"):
                for gender_config in ("match", "swap"):
                    out.append(_render_reflexive(
                        unique_id, item,
                        anaphor_type=anaphor_type,
                        gender_config=gender_config,
                        modifier_type=modifier_type,
                    ))
    return out


def _generate_principlec(
    num_items: int, lexicon: CCommandLexicon, seed: int,
) -> list[StimulusRecord]:
    pool = _enumerate_principlec_items(lexicon, seed=seed, shuffle=True)
    if not pool:
        raise ValueError("Principle C item pool is empty.")
    out: list[StimulusRecord] = []
    for idx in range(num_items):
        item = pool[idx % len(pool)]
        unique_id = item.item_id if num_items <= len(pool) else f"{item.item_id}-{idx:05d}"
        for cond in ("violation", "obviated"):
            out.append(_render_principlec(unique_id, item, cond))
    return out


def _generate_boundvar(
    num_items: int, lexicon: CCommandLexicon, seed: int,
) -> list[StimulusRecord]:
    pool = _enumerate_boundvar_items(lexicon, seed=seed, shuffle=True)
    if not pool:
        raise ValueError("Bound-variable item pool is empty.")
    out: list[StimulusRecord] = []
    for idx in range(num_items):
        item = pool[idx % len(pool)]
        unique_id = item.item_id if num_items <= len(pool) else f"{item.item_id}-{idx:05d}"
        for cond in ("subj_quant", "mod_quant"):
            out.append(_render_boundvar(unique_id, item, cond))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_subseq_after(
    haystack: list[str], needle: list[str], after_index: int,
) -> int | None:
    """Find ``needle`` in ``haystack`` at positions >= after_index."""
    if not needle:
        return None
    for i in range(max(0, after_index), len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def _require(value: int | None, role: str, text: str) -> int:
    if value is None:
        raise ValueError(
            f"Could not locate role {role!r} in tokens for: {text!r}. "
            "This usually indicates a template/lexicon mismatch."
        )
    return value


__all__ = [
    "BoundVarItem",
    "CCommandLexicon",
    "DEFAULT_LEXICON",
    "PrincipleCItem",
    "ReflexiveItem",
    "generate_c_command_stimuli",
]
