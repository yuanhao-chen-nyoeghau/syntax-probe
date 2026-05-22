"""Stimulus verification (Task 0.5).

Before running any probe experiment, verify the methodological precondition:
within each minimal pair / triple, the gold dependency-tree distance for the
experimental word pair must be (approximately) constant across conditions. If
it varies, the experimental contrast is contaminated by dependency-level
differences, undermining the interpretation of probe-distance differences.

This runner parses stimuli with spaCy or Stanza, computes the gold tree
distance for every (condition, word-pair-role) combo, and reports whether
distances are constant within items.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

from ..core.config import AppConfig, StimulusVerificationConfig
from ..core.context import RunContext, write_manifest
from ..core.io import write_json
from ..core.seed import seed_everything
from ..corpora.distance import compute_dependency_distance_matrix
from ..corpora.schema import DependencyArc, ParsedSentence
from ..stimuli.c_command import generate_c_command_stimuli
from ..stimuli.schema import StimulusRecord
from ..stimuli.wh_extraction import generate_wh_extraction_stimuli
from .base import ExperimentResult, ExperimentRunner
from .registry import register_experiment_runner

logger = logging.getLogger(__name__)

# Word-pair roles to inspect per experiment kind.
#
# For experiments with sub-experiments (currently only ``c_command``), the
# value is replaced by a per-record lookup in ``_pairs_for_record``.
_PAIRS_FOR_EXPERIMENT: dict[str, list[tuple[str, str]]] = {
    "wh_extraction": [
        ("wh", "embedded_verb"),
        ("wh", "embedded_subject"),
        ("embedded_subject", "embedded_verb"),
    ],
}

# Sub-experiment-specific pair tables for c_command stimuli verification.
_C_COMMAND_PAIRS_BY_SUBEXPERIMENT: dict[str, list[tuple[str, str]]] = {
    "reflexive":   [("subject", "anaphor"), ("modifier", "anaphor")],
    "principle_c": [("pronoun", "r_expression")],
    "bound_var":   [("quant_noun", "pronoun"), ("quantifier", "pronoun")],
}


def _group_key_for_record(
    target_experiment: str, record: StimulusRecord,
) -> tuple:
    """Return a tuple key identifying the group of records that should have
    the SAME UD distances by design.

    Records sharing a group key have *structurally equivalent* templates and
    only differ in lexical content (or other UD-irrelevant manipulations).
    Groups with > 1 record can be checked for within-group spread; groups
    with 1 record (e.g. for sub-experiments where every condition has its
    own structure) are reported but not checked for spread.

    For c_command:
      * reflexive   : group by (item_id, modifier_type). The 4 conditions
                      within a group (refl/pron x match/swap) are
                      structurally identical.
      * principle_c : group by (item_id, condition). Each group has 1
                      record because the two conditions
                      (violation, obviated) have different structures.
      * bound_var   : group by (item_id, condition). Same reason.

    For wh_extraction:
      * group by item_id. The three conditions (bare/inf/finite) are
        designed to give the same UD distance for the experimental pairs.

    The third element of the returned tuple is a string label used in the
    report rows so a human reader can tell what dimension the group key
    covers.
    """
    if target_experiment == "c_command":
        sub = record.metadata.get("subexperiment")
        if sub == "reflexive":
            return (record.item_id, record.metadata["modifier_type"], "modifier_type")
        # principle_c, bound_var: one record per (item_id, condition).
        return (record.item_id, record.condition, "condition")
    # wh_extraction (and any future experiment without sub-experiments).
    return (record.item_id, "<all>", "item")


def _pairs_for_record(
    target_experiment: str, record: StimulusRecord,
) -> list[tuple[str, str]]:
    if target_experiment == "c_command":
        sub = record.metadata.get("subexperiment")
        if sub is None:
            raise ValueError(
                f"c_command record {record.stimulus_id!r} missing 'subexperiment' metadata."
            )
        return _C_COMMAND_PAIRS_BY_SUBEXPERIMENT[sub]
    return _PAIRS_FOR_EXPERIMENT[target_experiment]


class StimulusVerificationRunner(ExperimentRunner):
    kind = "verify_stimuli"

    def run(self, *, app_config: AppConfig, context: RunContext) -> ExperimentResult:
        if not isinstance(app_config.experiment, StimulusVerificationConfig):
            raise TypeError("Expected verify_stimuli config")

        seed_everything(app_config.runtime.seed)
        cfg = app_config.experiment

        # Generate stimuli for the target experiment.
        if cfg.target_experiment == "wh_extraction":
            stimulus_set = generate_wh_extraction_stimuli(num_items=cfg.num_items)
        elif cfg.target_experiment == "c_command":
            stimulus_set = generate_c_command_stimuli(num_items=cfg.num_items)
        else:
            raise NotImplementedError(f"target_experiment={cfg.target_experiment!r}")

        # Parse every stimulus. We use the parser's batch interface so that
        # transformer-based parsers (en_core_web_trf, Stanza neural pipelines)
        # can amortise GPU launch cost across batches.
        parser = _make_parser(cfg)
        logger.info(
            "Parsing %d stimuli with %s (batch_size=%d, gpu_id=%s)",
            len(stimulus_set.records),
            parser.describe(),
            cfg.parser_batch_size,
            cfg.gpu_id,
        )
        parsed_records: list[tuple[StimulusRecord, ParsedSentence]] = []
        for record, parsed in parser.parse_batch(
            stimulus_set.records, batch_size=cfg.parser_batch_size,
        ):
            if parsed is None:
                logger.warning("Parser returned no parse for stimulus %s", record.stimulus_id)
                continue
            parsed_records.append((record, parsed))

        # For each *structural-equivalence group* (records that should have
        # the same UD distances by design), compute distances for each role
        # pair and check spread. Group keys are determined by
        # ``_group_key_for_record`` (per experiment / sub-experiment).
        # Groups with only 1 record record their distance(s) but are not
        # checked for spread (and so always satisfy ``ok=True``).
        per_group: dict[tuple, list[tuple[StimulusRecord, ParsedSentence]]] = defaultdict(list)
        for record, parsed in parsed_records:
            per_group[_group_key_for_record(cfg.target_experiment, record)].append((record, parsed))

        report_rows: list[dict[str, Any]] = []
        any_failure = False

        for group_key, group in per_group.items():
            item_id, group_value, group_dim = group_key
            distances_by_pair: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
            for record, parsed in group:
                pairs_to_check = _pairs_for_record(cfg.target_experiment, record)
                # Use parser's tokenization, not stimulus tokenization, for indexing.
                # We need to find each role's word in the parser's token list.
                role_token_indices = _resolve_role_indices_in_parsed(record, parsed)
                if role_token_indices is None:
                    continue
                matrix = compute_dependency_distance_matrix(parsed)
                for left_role, right_role in pairs_to_check:
                    if left_role not in role_token_indices or right_role not in role_token_indices:
                        continue
                    li = role_token_indices[left_role]
                    ri = role_token_indices[right_role]
                    distances_by_pair[(left_role, right_role)][record.condition] = int(
                        matrix[li, ri]
                    )

            for (left_role, right_role), per_condition in distances_by_pair.items():
                values = list(per_condition.values())
                if not values:
                    continue
                spread = max(values) - min(values)
                # If the group has only one condition, spread is trivially 0
                # and we don't enforce it (we just report the distance).
                if len(per_condition) >= 2:
                    ok = spread <= cfg.max_distance_difference
                    if not ok:
                        any_failure = True
                else:
                    ok = True
                report_rows.append(
                    {
                        "item_id": item_id,
                        "group_dim": group_dim,
                        "group_value": group_value,
                        "left_role": left_role,
                        "right_role": right_role,
                        "distances_by_condition": per_condition,
                        "spread": spread,
                        "ok": ok,
                    }
                )

        report_path = context.artifact_dir / "verification_report.json"
        write_json(
            report_path,
            {
                "target_experiment": cfg.target_experiment,
                "parser": cfg.parser,
                "spacy_model": cfg.spacy_model if cfg.parser == "spacy" else None,
                "gpu_id": cfg.gpu_id,
                "parser_batch_size": cfg.parser_batch_size,
                "max_distance_difference": cfg.max_distance_difference,
                "rows": report_rows,
                "any_failure": any_failure,
                "n_stimuli": len(stimulus_set.records),
                "n_parsed": len(parsed_records),
            },
        )

        manifest_path = write_manifest(
            context,
            {
                "any_failure": any_failure,
                "n_rows": len(report_rows),
                "artifacts": {"verification_report": str(report_path)},
            },
        )

        if any_failure:
            logger.warning(
                "Verification FAILED for some items. Inspect %s for details.", report_path
            )
        else:
            logger.info(
                "Verification PASSED: all %d (item, role-pair) entries have spread <= %d.",
                len(report_rows),
                cfg.max_distance_difference,
            )

        return ExperimentResult(
            experiment_kind=self.kind,
            run_id=context.run_id,
            summary={"any_failure": any_failure, "n_rows": len(report_rows)},
            artifacts={"verification_report": report_path, "manifest": manifest_path},
        )


# ---------------------------------------------------------------------------
# Parser adapters
# ---------------------------------------------------------------------------


class _ParserAdapter:
    """Interface for dependency parsers used by the verifier.

    Adapters expose two access patterns:

    * ``parse(record) -> ParsedSentence | None`` for one-off / test use.
    * ``parse_batch(records, batch_size) -> Iterator[(record, parsed)]`` for
      production use. Transformer-based parsers (en_core_web_trf, neural
      Stanza) are dramatically faster when batched, especially on GPU.
      Non-transformer parsers fall back to per-sentence calls internally
      (``batch_size`` is then a no-op).

    ``describe()`` returns a short human-readable identifier (e.g.
    ``"spacy/en_core_web_trf (gpu=0)"``) that is logged at run start.
    """

    def parse(self, record: StimulusRecord) -> ParsedSentence | None: ...

    def parse_batch(
        self,
        records: Sequence[StimulusRecord],
        *,
        batch_size: int,
    ) -> Iterator[tuple[StimulusRecord, ParsedSentence | None]]: ...

    def describe(self) -> str: ...


class _SpacyParser(_ParserAdapter):
    def __init__(self, model_name: str, *, gpu_id: int | None = None) -> None:
        try:
            import spacy
        except ImportError as err:
            raise ImportError(
                "spacy is required for stimulus verification (pip install spacy)"
            ) from err

        # GPU enable MUST happen before spacy.load so the loaded pipeline
        # places transformer weights on the right device. lg/md/sm models
        # are CNN-based and ignore GPU; trf is a roberta-base pipeline that
        # needs GPU for any reasonable wall-clock time.
        self._gpu_id = gpu_id
        self._on_gpu = False
        if gpu_id is not None:
            try:
                # spacy.require_gpu(gpu_id) errors if GPU isn't usable;
                # spacy.prefer_gpu(gpu_id) silently falls back to CPU.
                # We use require_gpu so misconfiguration fails loudly.
                spacy.require_gpu(gpu_id=gpu_id)
                self._on_gpu = True
            except Exception as err:
                raise RuntimeError(
                    f"spacy.require_gpu(gpu_id={gpu_id}) failed: {err}. "
                    f"Either set gpu_id=None to use CPU, or fix the GPU setup "
                    f"(install cupy / spacy[cuda12x] / etc., and check "
                    f"CUDA_VISIBLE_DEVICES)."
                ) from err
        try:
            self.nlp = spacy.load(model_name)
        except OSError as err:
            raise OSError(
                f"spaCy model {model_name!r} not installed. Run: "
                f"python -m spacy download {model_name}"
            ) from err
        self._model_name = model_name

    def describe(self) -> str:
        device = f"gpu={self._gpu_id}" if self._on_gpu else "cpu"
        return f"spacy/{self._model_name} ({device})"

    def parse(self, record: StimulusRecord) -> ParsedSentence | None:
        return self._doc_to_parsed(record, self.nlp(record.text))

    def parse_batch(
        self,
        records: Sequence[StimulusRecord],
        *,
        batch_size: int,
    ) -> Iterator[tuple[StimulusRecord, ParsedSentence | None]]:
        records = list(records)
        texts = [r.text for r in records]
        # spaCy's nlp.pipe yields docs in order; we zip them back to records.
        for record, doc in zip(records, self.nlp.pipe(texts, batch_size=batch_size)):
            try:
                parsed = self._doc_to_parsed(record, doc)
            except Exception as err:  # pragma: no cover (defensive)
                logger.warning("parse failed for %s: %s", record.stimulus_id, err)
                parsed = None
            yield record, parsed

    @staticmethod
    def _doc_to_parsed(record: StimulusRecord, doc: object) -> ParsedSentence | None:
        # ``doc`` is a spacy.tokens.Doc but we keep the type loose here so the
        # module can be imported without pulling spacy when only tests run.
        tokens = [tok.text for tok in doc]
        arcs: list[DependencyArc] = []
        for tok in doc:
            if tok.dep_ == "ROOT" or tok.head.i == tok.i:
                arcs.append(DependencyArc(head_index=-1, dependent_index=tok.i, relation="root"))
            else:
                arcs.append(
                    DependencyArc(
                        head_index=tok.head.i, dependent_index=tok.i, relation=tok.dep_
                    )
                )
        return ParsedSentence(
            sentence_id=record.stimulus_id,
            text=record.text,
            tokens=tokens,
            dependency_arcs=arcs,
            metadata={"parser": "spacy"},
        )


class _StanzaParser(_ParserAdapter):
    def __init__(self, *, gpu_id: int | None = None) -> None:
        try:
            import stanza
        except ImportError as err:
            raise ImportError("stanza is required (pip install stanza)") from err
        self._gpu_id = gpu_id
        # Stanza accepts use_gpu via Pipeline constructor; honour gpu_id.
        kwargs: dict[str, object] = {
            "lang": "en",
            "processors": "tokenize,pos,lemma,depparse",
            "use_gpu": gpu_id is not None,
        }
        if gpu_id is not None:
            kwargs["device"] = f"cuda:{gpu_id}"
        self.nlp = stanza.Pipeline(**kwargs)

    def describe(self) -> str:
        device = f"gpu={self._gpu_id}" if self._gpu_id is not None else "cpu"
        return f"stanza/en ({device})"

    def parse(self, record: StimulusRecord) -> ParsedSentence | None:
        return self._doc_to_parsed(record, self.nlp(record.text))

    def parse_batch(
        self,
        records: Sequence[StimulusRecord],
        *,
        batch_size: int,
    ) -> Iterator[tuple[StimulusRecord, ParsedSentence | None]]:
        # Stanza accepts a list of texts and processes them as separate
        # documents; this is its standard batched API.
        records = list(records)
        # Stanza's batching is configured globally on the pipeline; we batch
        # in chunks for memory predictability.
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            [{"text": r.text} for r in chunk]
            try:
                out_docs = self.nlp.bulk_process([r.text for r in chunk])
            except AttributeError:
                # Older Stanza versions: fall back to per-record.
                out_docs = [self.nlp(r.text) for r in chunk]
            for record, doc in zip(chunk, out_docs):
                try:
                    parsed = self._doc_to_parsed(record, doc)
                except Exception as err:  # pragma: no cover (defensive)
                    logger.warning("parse failed for %s: %s", record.stimulus_id, err)
                    parsed = None
                yield record, parsed

    @staticmethod
    def _doc_to_parsed(record: StimulusRecord, doc: object) -> ParsedSentence | None:
        if not doc.sentences:
            return None
        sentence = doc.sentences[0]
        tokens: list[str] = [w.text for w in sentence.words]
        arcs: list[DependencyArc] = []
        for word in sentence.words:
            dep_idx = word.id - 1  # stanza uses 1-based ids
            head = word.head
            if head == 0:
                arcs.append(
                    DependencyArc(head_index=-1, dependent_index=dep_idx, relation=word.deprel)
                )
            else:
                arcs.append(
                    DependencyArc(
                        head_index=head - 1, dependent_index=dep_idx, relation=word.deprel
                    )
                )
        return ParsedSentence(
            sentence_id=record.stimulus_id,
            text=record.text,
            tokens=tokens,
            dependency_arcs=arcs,
            metadata={"parser": "stanza"},
        )


def _make_parser(cfg: StimulusVerificationConfig) -> _ParserAdapter:
    if cfg.parser == "spacy":
        return _SpacyParser(cfg.spacy_model, gpu_id=cfg.gpu_id)
    if cfg.parser == "stanza":
        return _StanzaParser(gpu_id=cfg.gpu_id)
    raise ValueError(f"Unknown parser: {cfg.parser!r}")


def _resolve_role_indices_in_parsed(
    record: StimulusRecord, parsed: ParsedSentence
) -> dict[str, int] | None:
    """Locate each role's first token in the parser's token list.

    The stimulus stores role indices into its own token list, which is built
    with `tokenize_words`. The parser may tokenize slightly differently, so we
    re-locate each role by string match in the parser's tokens.
    """
    out: dict[str, int] = {}
    for role, stim_index in record.role_indices.items():
        if stim_index >= len(record.tokens):
            return None
        target = record.tokens[stim_index]
        # Find the first occurrence in parsed.tokens.
        try:
            out[role] = parsed.tokens.index(target)
        except ValueError:
            logger.debug(
                "Role %s (%r) not found in parsed tokens for %s: %s",
                role,
                target,
                record.stimulus_id,
                parsed.tokens,
            )
            return None
    return out


register_experiment_runner("verify_stimuli", StimulusVerificationRunner)


__all__ = ["StimulusVerificationRunner"]
