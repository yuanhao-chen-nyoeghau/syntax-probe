"""Run wh-extraction and c-command stats for all registered models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import c_command_stats, wh_extraction_stats
from ._common import add_model_selection_args


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--verify-run", type=Path, default=None,
                   help="Explicit verify_stimuli run (shared across models).")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--fit-method", choices=["ols_cr1", "mixedlm"], default="ols_cr1")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    add_model_selection_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Build the forwarded argv for each sub-script. We always pass
    # --all-models (or --include) since run_all_stats is only useful in
    # multi-model mode; if neither flag was given, default to --all-models.
    common: list[str] = [
        "--results-dir", str(args.results_dir),
        "--n-bootstrap", str(args.n_bootstrap),
        "--fit-method", args.fit_method,
        "--log-level", args.log_level,
    ]
    if args.verify_run is not None:
        common += ["--verify-run", str(args.verify_run)]
    if args.fail_fast:
        common += ["--fail-fast"]

    if args.include:
        common += ["--include", *args.include]
    else:
        common += ["--all-models"]

    rc = 0
    print("\n" + "=" * 64)
    print("  RUN ALL STATS — wh_extraction")
    print("=" * 64)
    rc |= wh_extraction_stats.main(common)

    print("\n" + "=" * 64)
    print("  RUN ALL STATS — c_command")
    print("=" * 64)
    rc |= c_command_stats.main(common)

    return rc


if __name__ == "__main__":
    sys.exit(main())
