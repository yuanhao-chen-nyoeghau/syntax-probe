"""Statistical analyses for the apply_probes outputs.

Each script in this package processes one or more apply_probes runs and
writes summary CSVs / JSON into the run directory. The shared utilities
for run discovery, cluster bootstrap, multiple-comparisons correction,
and clustered-regression fits live in ``_common``.

Entry points (also exposed as console scripts via pyproject.toml):

    python -m analyses.wh_extraction_stats   |  wh-stats
    python -m analyses.c_command_stats       |  cc-stats
    python -m analyses.run_all_stats         |  run-all-stats
"""
