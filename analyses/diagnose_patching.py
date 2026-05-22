"""Quick diagnostic for an activation_patching run's per_trial_predictions.jsonl."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def _summarize_deltas(deltas: list[float]) -> str:
    """One-line summary: n, mean, median, %positive."""
    if not deltas:
        return "n=0"
    n = len(deltas)
    pos = sum(1 for d in deltas if d > 0)
    return (
        f"n={n}, mean={mean(deltas):+.3f}, median={median(deltas):+.3f}, "
        f"%pos={100 * pos / n:.0f}%"
    )


def diagnose(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"=== {path.name} ===")
    print(f"Total trials: {len(rows)}")
    if not rows:
        return

    # Schema check on first row.
    expected = {
        "trial_id", "experiment_kind", "item_id",
        "source_stimulus_id", "target_stimulus_id",
        "intervention_role", "intervention_layer",
        "measurement_layers", "metadata",
        "patched_distances", "unpatched_distances",
    }
    actual = set(rows[0].keys())
    missing = expected - actual
    extra = actual - expected
    print(f"Schema: missing={sorted(missing) or 'OK'}, extra={sorted(extra) or 'OK'}")

    # Per-cell trial counts.
    cells: Counter[tuple[str, str | None, str | None]] = Counter()
    for r in rows:
        meta = r.get("metadata", {})
        cells[(
            r["intervention_role"],
            meta.get("source_condition") or meta.get("source_gender_config"),
            meta.get("target_condition") or meta.get("target_gender_config"),
        )] += 1
    print("\nPer-cell trial counts:")
    for cell, count in sorted(cells.items()):
        print(f"  {cell}: {count}")

    # Per-role pair selection.
    #
    # At ``L_patch = L_measure``, a probe-distance for pair (p1, p2) is
    # only affected by a patch at position p if p ∈ {p1, p2}; other
    # positions retain their unpatched residuals at the measurement
    # layer, so pair-distances not involving p are mathematical zeros.
    # Reporting Δβ on a pair whose endpoints don't include the patched
    # role gives a structural null rather than a true causal null, so we
    # walk the role-appropriate pairs explicitly here.
    role_to_pairs = {
        "wh": ("wh-esubj", "wh-evb"),
        "embedded_subject": ("wh-esubj", "esubj-evb"),
        "embedded_verb": ("wh-evb", "esubj-evb"),
        "anaphor": ("anaphor-subject", "anaphor-modifier"),
    }

    # Group rows by cell once; downstream loops just iterate the rows.
    rows_by_cell: dict[tuple[str, str | None, str | None], list[dict]] = defaultdict(list)
    for r in rows:
        meta = r.get("metadata", {})
        cell = (
            r["intervention_role"],
            meta.get("source_condition") or meta.get("source_gender_config"),
            meta.get("target_condition") or meta.get("target_gender_config"),
        )
        rows_by_cell[cell].append(r)

    for cell in sorted(rows_by_cell):
        role = cell[0]
        pairs = role_to_pairs.get(role)
        if pairs is None:
            # Unknown role: fall back to the first pair present in the
            # first row, so the report stays useful for new experiments.
            pairs = (next(iter(rows_by_cell[cell][0]["patched_distances"])),)
        print(f"\n=== cell {cell} (role-relevant pairs: {pairs}) ===")
        for pair in pairs:
            _report_cell_pair(rows_by_cell[cell], pair)

    # Sanity check: any trials where source == target stimulus_id?
    self_patches = sum(
        1 for r in rows
        if r["source_stimulus_id"] == r["target_stimulus_id"]
    )
    print(f"\nSelf-patches (source==target, should be 0): {self_patches}")


def _report_cell_pair(cell_rows: list[dict], pair: str) -> None:
    """Print Δβ stats, quantile distribution, and top outliers for one
    (cell, pair) combination."""
    deltas: list[float] = []
    patched_vals: list[float] = []
    unpatched_vals: list[float] = []
    triples: list[tuple[float, float, float, dict]] = []  # (Δβ, p, u, row)
    missing = 0
    for r in cell_rows:
        L = r["intervention_layer"]
        try:
            patched = float(r["patched_distances"][pair][str(L)])
            unpatched = float(r["unpatched_distances"][pair][str(L)])
        except (KeyError, TypeError):
            missing += 1
            continue
        d = patched - unpatched
        deltas.append(d)
        patched_vals.append(patched)
        unpatched_vals.append(unpatched)
        triples.append((d, patched, unpatched, r))
    print(f"  pair={pair}:")
    if missing:
        print(f"    ({missing} trials missing {pair} at intervention layer)")
    print(f"    Δβ:        {_summarize_deltas(deltas)}")
    print(f"    patched:   {_summarize_deltas(patched_vals)}")
    print(f"    unpatched: {_summarize_deltas(unpatched_vals)}")
    if deltas:
        # Quantile snapshot — surfaces heavy tails the mean/median pair
        # can hide.
        qs = _quantiles(deltas, (0.01, 0.05, 0.50, 0.95, 0.99))
        print(
            "    Δβ quantiles: "
            f"p01={qs[0]:+.3f}  p05={qs[1]:+.3f}  p50={qs[2]:+.3f}  "
            f"p95={qs[3]:+.3f}  p99={qs[4]:+.3f}  "
            f"min={min(deltas):+.3f}  max={max(deltas):+.3f}"
        )
        # Top-3 outliers by |Δβ| with identifying stimulus ids.
        outliers = sorted(triples, key=lambda t: abs(t[0]), reverse=True)[:3]
        if outliers and abs(outliers[0][0]) > 1e-6:
            print("    Top outliers (by |Δβ|):")
            for d, p, u, r in outliers:
                print(
                    f"      Δβ={d:+10.3f}  patched={p:9.3f}  unpatched={u:9.3f}"
                    f"  src={r['source_stimulus_id']}  tgt={r['target_stimulus_id']}"
                )


def _quantiles(xs: list[float], qs: tuple[float, ...]) -> list[float]:
    """Linear-interp quantiles (numpy-free; tiny enough not to need a dep)."""
    if not xs:
        return [float("nan")] * len(qs)
    s = sorted(xs)
    n = len(s)
    out: list[float] = []
    for q in qs:
        if n == 1:
            out.append(s[0])
            continue
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append(s[lo] * (1 - frac) + s[hi] * frac)
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-per_trial_predictions.jsonl>",
              file=sys.stderr)
        sys.exit(1)
    diagnose(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
