"""Phase A: border-filter A/B test with paired bootstrap deltas."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from biohub.config import Config
from biohub.data import list_datasets
from biohub.tuning import bootstrap_ci, paired_bootstrap_delta, score_config_on_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    names = list_datasets(args.train_dir)
    base = Config().competition_v10_dog_preset()
    arms = {
        "baseline_q0.80": base,
        "q0.50": base.copy_with(border_keep_quantile=0.50),
        "q0.20": base.copy_with(border_keep_quantile=0.20),
        "q0.00": base.copy_with(border_keep_quantile=0.0),
        "border_yx1": base.copy_with(border_yx=1),
        "border_z1": base.copy_with(border_z=1),
        "q0.00_yx1_z1": base.copy_with(border_keep_quantile=0.0, border_yx=1, border_z=1),
    }
    values_by_arm = {}
    details = []
    for arm, cfg in arms.items():
        values = []
        for name in names:
            result = score_config_on_sample(args.train_dir, name, cfg, args.frames)
            values.append(result["score"])
            details.append({"arm": arm, "dataset": name, **result})
            print(arm, name, result, flush=True)
        values_by_arm[arm] = np.asarray(values, dtype=float)
    baseline = values_by_arm["baseline_q0.80"]
    rows = []
    for arm, values in values_by_arm.items():
        mean, low, high = bootstrap_ci(values)
        delta, delta_low, delta_high = paired_bootstrap_delta(baseline, values)
        rows.append({
            "arm": arm,
            "mean_score": mean,
            "ci_low": low,
            "ci_high": high,
            "paired_delta": delta,
            "paired_delta_ci_low": delta_low,
            "paired_delta_ci_high": delta_high,
            "accept": bool(arm != "baseline_q0.80" and delta_low > 0.0),
        })
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(details).to_csv("results/border_ab_test_phase_a_details.csv", index=False)
    summary = pd.DataFrame(rows).sort_values("mean_score", ascending=False)
    summary.to_csv("results/border_ab_test_phase_a.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
