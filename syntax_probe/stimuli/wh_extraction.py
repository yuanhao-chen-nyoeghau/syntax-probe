"""Wh-extraction stimulus generator.

Generates minimal triples contrasting wh-extraction across complement sizes:

    bare:        "What did she see him eat?"            (small clause)
    infinitival: "What did she expect him to eat?"      (infinitival TP)
    finite:      "What did she think he ate?"           (full CP)

The three conditions vary only the matrix verb and embedded clause type; the
wh-phrase, embedded subject, and embedded verb are held constant. The resulting
dependency parse should be (approximately) identical across conditions — this is
the precondition verified in the ``verify-stimuli`` step.

=============================================================================
Linguistic basis for verb classifications (see design_decisions.md)
=============================================================================

Bare small-clause condition
---------------------------
Perception verbs (``see``, ``watch``) and causative verbs (``make``, ``let``)
license a ``[NP VP_bare_inf]`` small clause (SC) complement (Stowell 1981;
Haegeman & Guéron 1999; Wikipedia: Small clause). Both sub-types work with any
transitive embedded verb, making them suitable for free combinatorial crossing.

``hear`` is excluded from the combinatorial pool: it carries an auditory
selectional restriction incompatible with most of our embedded verbs (``*She
heard him write the report`` is odd). ``have``-causative is also excluded
because the surface string ``did have him`` risks auxiliary-parsing artefacts.

Infinitival condition
---------------------
Two sub-types both produce the surface ``[V NP to VP_inf]`` (Wikipedia:
Exceptional case-marking; Wikipedia: Control (linguistics)):

* True ECM verbs: the embedded subject undergoes A-movement to the matrix
  object position and bears no thematic relation to the matrix verb.
  Representative: ``expect``.
* Object-control verbs: the matrix verb semantically selects its NP object as
  the controller of PRO in the embedded infinitival.
  Representatives: ``want``, ``allow``, ``convince``, ``require``.

For the wh-extraction experiment both sub-types are appropriate: the structural
depth of the complement (infinitival TP, no CP phase) is what distinguishes this
condition from the finite condition.

Finite condition
----------------
Bridge verbs (``think``, ``believe``, ``claim``, ``say``, ``know``,
``suppose``, ``report``) license a tensed CP complement. This is the deepest
structural environment: the wh-element has been extracted across a full CP phase
boundary. ``believe`` is used exclusively in the bridge-verb slot so that no
item ever repeats the same verb across slots.

Embedded transitive verbs
-------------------------
The wh-word ``what`` fills the direct-object gap of the embedded verb. All 20
embedded verbs are transitive with inanimate-compatible objects and have no
surface overlap with any matrix verb set.

=============================================================================
Combinatorial generation
=============================================================================

``enumerate_all_items`` crosses all lexical dimensions and applies two filters:

1. The matrix-subject nominative must differ from the embedded-subject nominative
   (prevents ``*What did she see she eat?``).
2. The three matrix verb slots must all be distinct within an item.

The resulting item pool is ~128 K entries for the default lexicon. A shuffled
seed-controlled slice is used by ``generate_wh_extraction_stimuli``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from ..utils.text import find_subsequence, tokenize_words
from .schema import StimulusRecord, StimulusSet

Condition = str  # "bare" | "infinitival" | "finite"
_CONDITION_RANKS: dict[str, int] = {"bare": 1, "infinitival": 2, "finite": 3}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WhExtractionItem:
    """A lexical item realized in three syntactic conditions.

    Case marking:
        ``embedded_subject``            — accusative; used in bare (SC) and
                                          infinitival (ECM/control) conditions.
        ``embedded_subject_nominative`` — nominative; used in finite (CP)
                                          condition.
        For DPs without case morphology ("the student"), both fields are equal.
    """

    item_id: str
    subject: str
    """Matrix clause subject (nominative), e.g. "she", "the teacher"."""
    embedded_subject: str
    """Accusative embedded subject; used in bare and infinitival conditions."""
    embedded_subject_nominative: str
    """Nominative embedded subject; used in finite condition."""
    bare_matrix_verb: str
    """Perception or causative verb licensing a bare small clause."""
    infinitival_matrix_verb: str
    """ECM or object-control verb licensing an infinitival complement."""
    finite_matrix_verb: str
    """Bridge verb licensing a finite CP complement."""
    embedded_verb_base: str
    """Base/infinitival form of the embedded transitive verb."""
    embedded_verb_past: str
    """Past-tense form; used in the finite condition."""


@dataclass(frozen=True, slots=True)
class WhExtractionLexicon:
    """Full lexical inventory for combinatorial item generation.

    All string sequences are tuples so the lexicon is hashable and immutable.
    """

    subjects: tuple[str, ...]
    """Matrix clause subjects (nominative form)."""

    embedded_subject_pairs: tuple[tuple[str, str], ...]
    """(accusative_form, nominative_form) pairs for embedded subjects."""

    bare_sc_verbs: tuple[str, ...]
    """Perception and causative verbs licensing bare small clauses."""

    infinitival_verbs: tuple[tuple[str, str], ...]
    """(verb, verb_class) pairs where verb_class is "ecm" or "obj_control"."""

    bridge_verbs: tuple[str, ...]
    """Bridge verbs licensing finite CP complements."""

    embedded_verbs: tuple[tuple[str, str], ...]
    """(base_form, past_tense_form) pairs for embedded transitive verbs."""


# ---------------------------------------------------------------------------
# Default lexicon
# ---------------------------------------------------------------------------

DEFAULT_LEXICON: WhExtractionLexicon = WhExtractionLexicon(
    # 7 matrix subjects — single pronouns and role-DPs compatible with all
    # bare SC verbs (no agreement issues since aux is always "did").
    subjects=(
        "she",
        "he",
        "they",
        "I",
        "the teacher",
        "the manager",
        "the detective",
    ),
    # 7 embedded subject pairs (accusative → nominative).
    # Pronoun pairs differ in case; DP pairs repeat the same string.
    embedded_subject_pairs=(
        ("him", "he"),
        ("her", "she"),
        ("us", "we"),
        ("them", "they"),
        ("the student",   "the student"),
        ("the assistant", "the assistant"),
        ("the suspect",   "the suspect"),
    ),
    # 4 bare SC verbs — perception (see, watch) and causative (make, let).
    # All freely combine with every embedded verb below.
    # Excluded: hear (auditory selectional restriction), have (auxiliary clash).
    bare_sc_verbs=(
        "see",    # visual perception
        "watch",  # visual perception, sustained
        "make",   # causative
        "let",    # permissive causative
    ),
    # 4 infinitival verbs — (verb, class) pairs.
    # ECM: subject raises to matrix object position (Chomsky 1986).
    # obj_control: matrix object controls PRO (Radford 1988).
    #
    # Removed at scale verification (1000 items, May 2026):
    # - "convince": spaCy gives NP a direct-object arc to "convince" rather
    #   than treating it as embedded subject → dep-path distances of 3 where
    #   design expects 1-2. Caused 20 spread=2 failures.
    # - "require": same problem (NP as semantic object of "require"). 17 failures.
    # "need" included instead: spaCy parses "need NP to VP" consistently with
    # "want NP to VP", giving stable dep distances across conditions.
    infinitival_verbs=(
        ("expect", "ecm"),
        ("want",   "obj_control"),
        ("allow",  "obj_control"),
        ("need",   "obj_control"),
    ),
    # 7 bridge verbs licensing finite CP complements.
    # "believe" is placed here (not in the infinitival slot) to prevent
    # same-verb conflicts: it is a valid ECM verb in other contexts but its
    # primary use in everyday speech is as a bridge verb.
    bridge_verbs=(
        "think",    # most frequent English bridge verb
        "believe",  # bridge / ECM-ambiguous; bridge slot only
        "claim",    # assertive
        "say",      # neutral assertion
        "know",     # factive
        "suppose",  # tentative assertion
        "report",   # evidential
    ),
    # 20 embedded transitive verbs (base, past).
    # All take inanimate-compatible objects (compatible with wh-word "what").
    # No surface overlap with any matrix verb set.
    # Mix of regular and irregular past-tense forms for morphological variety.
    embedded_verbs=(
        ("eat",   "ate"),
        ("buy",   "bought"),
        ("read",  "read"),
        ("write", "wrote"),
        ("build", "built"),
        ("fix",   "fixed"),
        ("find",  "found"),
        ("clean", "cleaned"),
        ("cook",  "cooked"),
        ("sell",  "sold"),
        ("carry", "carried"),
        ("paint", "painted"),
        ("break", "broke"),
        ("steal", "stole"),
        ("bring", "brought"),
        ("open",  "opened"),
        ("send",  "sent"),
        ("drop",  "dropped"),
        ("use",   "used"),
        ("take",  "took"),
    ),
)


# ---------------------------------------------------------------------------
# Legacy hand-curated items — kept for backward compat with tests
# ---------------------------------------------------------------------------
# Verification results (May 2026, spaCy en_core_web_lg):
#   wh-L001: all spread=0 (parser-clean)
#   wh-L002: spread=1 on wh-pairs (infinitival shorter); known parser quirk,
#            direction works against experimental prediction so doesn't inflate
#            Type I error.
#   wh-L003: all spread=0 (parser-clean)
#   wh-L004: all spread=0 (parser-clean)
#   wh-L005: spread=1 on wh-pairs (same pattern as wh-L002); added to ensure
#            lexicon has multiple ECM-style pronoun-pair items.

LEGACY_ITEMS: tuple[WhExtractionItem, ...] = (
    WhExtractionItem(
        item_id="wh-L001",
        subject="she",
        embedded_subject="him",
        embedded_subject_nominative="he",
        bare_matrix_verb="see",
        infinitival_matrix_verb="expect",
        finite_matrix_verb="think",
        embedded_verb_base="eat",
        embedded_verb_past="ate",
    ),
    WhExtractionItem(
        item_id="wh-L002",
        subject="he",
        embedded_subject="her",
        embedded_subject_nominative="she",
        bare_matrix_verb="watch",
        infinitival_matrix_verb="want",
        finite_matrix_verb="believe",
        embedded_verb_base="buy",
        embedded_verb_past="bought",
    ),
    WhExtractionItem(
        item_id="wh-L003",
        subject="the teacher",
        embedded_subject="the student",
        embedded_subject_nominative="the student",
        bare_matrix_verb="let",
        infinitival_matrix_verb="allow",
        finite_matrix_verb="say",
        embedded_verb_base="read",
        embedded_verb_past="read",
    ),
    WhExtractionItem(
        item_id="wh-L004",
        subject="the manager",
        embedded_subject="the intern",
        embedded_subject_nominative="the intern",
        bare_matrix_verb="make",
        infinitival_matrix_verb="expect",
        finite_matrix_verb="claim",
        embedded_verb_base="write",
        embedded_verb_past="wrote",
    ),
    WhExtractionItem(
        item_id="wh-L005",
        subject="they",
        embedded_subject="us",
        embedded_subject_nominative="we",
        bare_matrix_verb="see",
        infinitival_matrix_verb="expect",
        finite_matrix_verb="claim",
        embedded_verb_base="visit",
        embedded_verb_past="visited",
    ),
)


# ---------------------------------------------------------------------------
# Combinatorial item generator
# ---------------------------------------------------------------------------


def enumerate_all_items(
    lexicon: WhExtractionLexicon = DEFAULT_LEXICON,
    *,
    seed: int = 0,
    shuffle: bool = True,
) -> list[WhExtractionItem]:
    """Generate all valid ``WhExtractionItem`` instances from a lexicon.

    Validity constraints applied:
    1. The matrix subject (nominative) must differ from the embedded subject
       nominative, preventing ``*What did she see she eat?``
    2. The three matrix verb slots must all be distinct within an item.

    The resulting list is deterministically shuffled when ``shuffle=True`` so
    that ``generate_wh_extraction_stimuli(num_items=N)`` produces a diverse
    sample irrespective of N.

    With the default lexicon the valid pool contains ~128 K items; a slice of
    the first N is used by ``generate_wh_extraction_stimuli``.

    Args:
        lexicon: lexical inventory to cross. Defaults to ``DEFAULT_LEXICON``.
        seed: random seed for the shuffle. Ignored when ``shuffle=False``.
        shuffle: if True, shuffle the output deterministically.

    Returns:
        List of ``WhExtractionItem`` instances, one per valid combination.
    """
    all_items: list[WhExtractionItem] = []
    index = 0

    # Flatten infinitival_verbs to just the verb strings for loop iteration;
    # store the class label in the metadata via item_id suffix.
    inf_verbs = [(v, cls) for v, cls in lexicon.infinitival_verbs]

    for subj in lexicon.subjects:
        for emb_acc, emb_nom in lexicon.embedded_subject_pairs:
            # Constraint 1: matrix subject ≠ embedded subject (nominative forms).
            if subj == emb_nom:
                continue

            for bare_v in lexicon.bare_sc_verbs:
                for inf_v, _inf_cls in inf_verbs:
                    for fin_v in lexicon.bridge_verbs:
                        # Constraint 2: all three matrix verb slots distinct.
                        if len({bare_v, inf_v, fin_v}) < 3:
                            continue

                        for emb_base, emb_past in lexicon.embedded_verbs:
                            # Assign a stable, human-readable item id that encodes
                            # the lexical content without being a numeric index.
                            # Format: wh-BARE-INF-FIN-EMBD-SUBJ where each
                            # component is abbreviated to 3 chars.
                            item_id = (
                                f"wh-{bare_v[:3]}-{inf_v[:3]}-{fin_v[:3]}"
                                f"-{emb_base[:3]}-{index:05d}"
                            )
                            all_items.append(
                                WhExtractionItem(
                                    item_id=item_id,
                                    subject=subj,
                                    embedded_subject=emb_acc,
                                    embedded_subject_nominative=emb_nom,
                                    bare_matrix_verb=bare_v,
                                    infinitival_matrix_verb=inf_v,
                                    finite_matrix_verb=fin_v,
                                    embedded_verb_base=emb_base,
                                    embedded_verb_past=emb_past,
                                )
                            )
                            index += 1

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_items)

    return all_items


# ---------------------------------------------------------------------------
# Stimulus set generator
# ---------------------------------------------------------------------------


def generate_wh_extraction_stimuli(
    num_items: int = 1000,
    *,
    items: Sequence[WhExtractionItem] | None = None,
    lexicon: WhExtractionLexicon = DEFAULT_LEXICON,
    seed: int = 0,
) -> StimulusSet:
    """Produce a ``StimulusSet`` of minimal triples for wh-extraction.

    Two usage modes:

    **Combinatorial (default):** When ``items`` is ``None``, items are drawn
    from the full combinatorial enumeration of ``lexicon``. With the default
    lexicon (~103 K valid items after removing convince/require) and
    ``num_items=1000``, this produces 3,000 diverse stimuli with no repetition.

    **Explicit (for tests / reproducibility):** Pass ``items`` directly to
    produce stimuli from a known list, cycling if ``num_items > len(items)``.
    This is the right approach for unit tests that assert specific token
    sequences.

    Args:
        num_items: number of base items. Total stimuli = ``3 * num_items``.
            Default is 1000, producing 3,000 stimuli from the combinatorial pool.
        items: explicit item list. If ``None``, uses the combinatorial pool.
        lexicon: lexicon to enumerate from (ignored when ``items`` is given).
        seed: random seed for the combinatorial shuffle.

    Raises:
        ValueError: if ``num_items < 1``.
    """
    if num_items < 1:
        raise ValueError("num_items must be at least 1")

    pool = enumerate_all_items(lexicon, seed=seed, shuffle=True) if items is None else list(items)

    if not pool:
        raise ValueError("Item pool is empty — check lexicon validity constraints.")

    records: list[StimulusRecord] = []
    for index in range(num_items):
        item = pool[index % len(pool)]
        unique_item_id = f"{item.item_id}"
        # When cycling (num_items > len(pool)), add a suffix so item_ids stay unique.
        if num_items > len(pool):
            unique_item_id = f"{item.item_id}-{index:04d}"
        for condition in ("bare", "infinitival", "finite"):
            records.append(_render_one(unique_item_id, item, condition))

    return StimulusSet(
        name="wh_extraction",
        experiment="wh_extraction",
        records=records,
        description=(
            "Wh-extraction minimal triples: bare SC / infinitival TP / finite CP."
        ),
    )


# ---------------------------------------------------------------------------
# Internal rendering
# ---------------------------------------------------------------------------


def _render_one(
    unique_item_id: str,
    item: WhExtractionItem,
    condition: Condition,
) -> StimulusRecord:
    """Render one stimulus for one condition, computing role indices."""
    if condition == "bare":
        embedded_subject_surface = item.embedded_subject
        text = (
            f"What did {item.subject} {item.bare_matrix_verb} "
            f"{embedded_subject_surface} {item.embedded_verb_base}?"
        )
        matrix_verb = item.bare_matrix_verb
        embedded_verb_surface = item.embedded_verb_base

    elif condition == "infinitival":
        embedded_subject_surface = item.embedded_subject
        text = (
            f"What did {item.subject} {item.infinitival_matrix_verb} "
            f"{embedded_subject_surface} to {item.embedded_verb_base}?"
        )
        matrix_verb = item.infinitival_matrix_verb
        embedded_verb_surface = item.embedded_verb_base

    elif condition == "finite":
        embedded_subject_surface = item.embedded_subject_nominative
        text = (
            f"What did {item.subject} {item.finite_matrix_verb} "
            f"{embedded_subject_surface} {item.embedded_verb_past}?"
        )
        matrix_verb = item.finite_matrix_verb
        embedded_verb_surface = item.embedded_verb_past

    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    tokens = tokenize_words(text)
    canonical_text = " ".join(tokens)

    # Compute role indices via subsequence search — robust to multi-word
    # subjects and the extra "to" token in the infinitival condition.
    role_indices: dict[str, int] = {}
    role_indices["wh"] = _require(
        find_subsequence(tokens, ["What"]), "wh", text
    )
    role_indices["matrix_verb"] = _require(
        find_subsequence(tokens, tokenize_words(matrix_verb)), "matrix_verb", text
    )
    role_indices["embedded_subject"] = _require(
        find_subsequence(tokens, tokenize_words(embedded_subject_surface)),
        "embedded_subject",
        text,
    )
    role_indices["embedded_verb"] = _require(
        find_subsequence_after(
            tokens,
            tokenize_words(embedded_verb_surface),
            role_indices["embedded_subject"],
        ),
        "embedded_verb",
        text,
    )

    return StimulusRecord(
        stimulus_id=f"{unique_item_id}-{condition}",
        item_id=unique_item_id,
        condition=condition,
        text=canonical_text,
        tokens=tokens,
        role_indices=role_indices,
        metadata={
            "condition_rank": _CONDITION_RANKS[condition],
            "subject": item.subject,
            "embedded_subject_surface": embedded_subject_surface,
            "matrix_verb": matrix_verb,
            "embedded_verb_surface": embedded_verb_surface,
            "lexical_item_id": item.item_id,
            "rendered_text": text,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_subsequence_after(
    haystack: list[str], needle: list[str], after_index: int
) -> int | None:
    """Find ``needle`` in ``haystack`` at positions ``>= after_index``."""
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
            "This usually indicates a template / lexicon mismatch."
        )
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Backward-compatible alias so existing code that imported ``DEFAULT_ITEMS``
#: still works. For new code, use ``LEGACY_ITEMS`` directly.
DEFAULT_ITEMS = LEGACY_ITEMS

__all__ = [
    "DEFAULT_ITEMS",
    "DEFAULT_LEXICON",
    "LEGACY_ITEMS",
    "WhExtractionItem",
    "WhExtractionLexicon",
    "enumerate_all_items",
    "find_subsequence_after",
    "generate_wh_extraction_stimuli",
]
