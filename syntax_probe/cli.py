"""Command-line interface.

Provides three subcommands:
    syntax-probe download-corpus               # Download UD EWT and convert to ParsedSentence JSONL
    syntax-probe run <config1.yaml> [<config2.yaml> ...]  # Run one or more experiments
    syntax-probe info                          # Show registered experiment kinds
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

# Importing experiments registers all runners.
from . import experiments  # noqa: F401
from .core.config import load_app_config
from .core.context import build_run_context
from .corpora.schema import ParsedCorpus
from .corpora.ud_ewt import UD_EWT_VERSION, download_ud_ewt, load_ud_ewt_split
from .experiments.registry import get_experiment_runner


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
def main() -> None:
    """syntax-probe — probing LLMs for formal syntactic structure."""


@main.command("download-corpus")
@click.option(
    "--output", "output_dir", type=click.Path(path_type=Path), default=Path("data/ud_ewt"),
    help="Where to put the raw .conllu files and the parsed JSONL.",
)
@click.option("--max-length", type=int, default=60, help="Drop sentences longer than this.")
@click.option("--min-length", type=int, default=3, help="Drop sentences shorter than this.")
@click.option("--log-level", default="INFO")
def download_corpus(
    output_dir: Path, max_length: int, min_length: int, log_level: str
) -> None:
    """Download UD English EWT and convert to a unified ParsedSentence JSONL."""
    _setup_logging(log_level)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = download_ud_ewt(output_dir / "raw")

    all_sentences = []
    for split, path in raw_paths.items():
        all_sentences.extend(
            load_ud_ewt_split(path, split=split, min_length=min_length, max_length=max_length)
        )

    corpus = ParsedCorpus(
        name=f"ud_ewt_{UD_EWT_VERSION}",
        sentences=all_sentences,
        source=f"UD_English-EWT@{UD_EWT_VERSION}",
    )

    parsed_path = output_dir / f"ud_ewt_{UD_EWT_VERSION}.jsonl"
    import json

    with parsed_path.open("w", encoding="utf-8") as f:
        for sentence in corpus.sentences:
            f.write(json.dumps(sentence.model_dump(mode="json")))
            f.write("\n")

    click.echo(f"Wrote {len(corpus.sentences)} parsed sentences to {parsed_path}")
    click.echo(f"  Train: {len(corpus.by_split('train'))}")
    click.echo(f"  Dev:   {len(corpus.by_split('dev'))}")
    click.echo(f"  Test:  {len(corpus.by_split('test'))}")


@main.command("run")
@click.argument(
    "config_paths",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--set", "overrides", multiple=True, metavar="KEY.PATH=VALUE",
    help=(
        "Override a config field at runtime, e.g. "
        "`--set experiment.probe_run_dir=results/probe_training/foo` or "
        "`--set runtime.seed=42`. Applied to *every* config in the batch. "
        "May be passed multiple times. Values are JSON-decoded when possible "
        "(numbers, bools, lists, null) and otherwise treated as plain strings; "
        "pydantic coerces to the schema-declared type."
    ),
)
@click.option(
    "--fail-fast", is_flag=True, default=False,
    help=(
        "Abort the batch on the first config that fails. Default behaviour is to "
        "continue with the remaining configs and report all failures at the end "
        "(useful when independent runs are batched, since one bad config "
        "shouldn't waste hours of compute)."
    ),
)
def run_experiment(
    config_paths: tuple[Path, ...],
    overrides: tuple[str, ...],
    fail_fast: bool,
) -> None:
    """Run one or more experiments given YAML config(s).

    Configs are processed sequentially. Pass several to batch them::

        syntax-probe run configs/probe_training/qwen3_8b_ud_ewt.yaml
        syntax-probe run configs/applications/wh_extraction_*.yaml
        syntax-probe run cfg1.yaml cfg2.yaml --set runtime.seed=42

    By default, a single config failure does not abort the rest of the
    batch; pass ``--fail-fast`` to override. Exit code is 0 only if every
    config in the batch succeeded.
    """
    successes: list[tuple[Path, str, Path]] = []
    failures: list[tuple[Path, str]] = []
    n = len(config_paths)

    # Logging is configured by the first config that loads successfully.
    # Subsequent configs' ``runtime.log_level`` is informational only --
    # ``logging.basicConfig`` is a no-op once handlers exist.
    logging_configured = False

    for i, config_path in enumerate(config_paths, start=1):
        if n > 1:
            click.echo(f"\n{'=' * 70}")
            click.echo(f"[{i}/{n}] {config_path}")
            click.echo(f"{'=' * 70}")
        try:
            app_config = load_app_config(config_path, overrides=list(overrides))
            if not logging_configured:
                _setup_logging(app_config.runtime.log_level)
                logging_configured = True
            context = build_run_context(app_config=app_config, config_path=config_path)

            runner = get_experiment_runner(app_config.experiment.kind)
            result = runner.run(app_config=app_config, context=context)

            click.echo(f"\nCompleted {result.experiment_kind} run {result.run_id}")
            click.echo(f"Run dir: {context.run_dir}")
            if result.artifacts:
                click.echo("Artifacts:")
                for name, path in result.artifacts.items():
                    click.echo(f"  {name}: {path}")
            successes.append((config_path, result.run_id, context.run_dir))
        except Exception as exc:  # noqa: BLE001  -- we want to keep going on any failure
            click.echo(
                f"\nFAILED {config_path}: {type(exc).__name__}: {exc}",
                err=True,
            )
            failures.append((config_path, f"{type(exc).__name__}: {exc}"))
            if fail_fast:
                click.echo("Aborting due to --fail-fast.", err=True)
                break

    # Per-batch summary (skip if a single config and it succeeded -- the
    # detailed per-config output already says everything useful in that case).
    if n > 1 or failures:
        click.echo(f"\n{'=' * 70}")
        click.echo(
            f"BATCH SUMMARY: {len(successes)} succeeded, {len(failures)} failed"
        )
        click.echo(f"{'=' * 70}")
        if successes:
            click.echo(f"\nSucceeded ({len(successes)}):")
            for cfg, _run_id, run_dir in successes:
                click.echo(f"  {cfg.name} -> {run_dir}")
        if failures:
            click.echo(f"\nFailed ({len(failures)}):")
            for cfg, err in failures:
                click.echo(f"  {cfg.name}: {err}")

    if failures:
        sys.exit(1)


@main.command("info")
def info() -> None:
    """Show registered experiment kinds and their config classes."""
    from .experiments.registry import _REGISTRY  # noqa: PLC0415

    click.echo("Registered experiment kinds:")
    for kind in sorted(_REGISTRY):
        click.echo(f"  {kind}")


if __name__ == "__main__":
    sys.exit(main())
