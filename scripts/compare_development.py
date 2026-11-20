#!/usr/bin/env python3
"""Compare canonically identified lineage-event tables without guessing identities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub.comparative_development import compare_lineage_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_events", type=Path)
    parser.add_argument("observed_events", type=Path)
    parser.add_argument("--time-tolerance-s", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/development_comparison"))
    args = parser.parse_args()
    reference = pd.read_csv(args.reference_events)
    observed = pd.read_csv(args.observed_events)
    divergences = compare_lineage_events(reference, observed, time_tolerance_s=args.time_tolerance_s)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    divergences.to_csv(args.output_dir / "divergences.csv", index=False)
    summary = {
        "reference": str(args.reference_events.resolve()),
        "observed": str(args.observed_events.resolve()),
        "time_tolerance_s": args.time_tolerance_s,
        "divergence_count": int(len(divergences)),
        "first_divergence_time_s": float(divergences["first_divergence_time_s"].min()) if len(divergences) else None,
        "counts_by_type": divergences["divergence_type"].value_counts().to_dict(),
        "interpretation": "Event differences require biological review; they are not a genetic diagnosis.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
