"""Phase A: classical-versus-DoG A/B test with the corrected evaluator."""

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
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    names = list_datasets(args.train_dir)[: args.limit]
    arms = {
        "V4_classical": Config().competition_v4_preset(),
        "V10_DoG": Config().competition_v10_dog_preset(),
    }
    rows = []
    scores_by_arm = {}
    for arm, cfg in arms.items():
        values = []
        for name in names:
            result = score_config_on_sample(args.train_dir, name, cfg, args.frames)
            values.append(result["score"])
            rows.append({"arm": arm, "dataset": name, **result})
            print(arm, name, result, flush=True)
        scores_by_arm[arm] = np.asarray(values, dtype=float)
    baseline = scores_by_arm["V4_classical"]
    candidate = scores_by_arm["V10_DoG"]
    mean_delta, delta_low, delta_high = paired_bootstrap_delta(baseline, candidate)
    summary = []
    for arm, values in scores_by_arm.items():
        mean, low, high = bootstrap_ci(values)
        summary.append({"arm": arm, "mean_score": mean, "ci_low": low, "ci_high": high})
    summary_df = pd.DataFrame(summary)
    summary_df["paired_delta_vs_V4"] = [0.0, mean_delta]
    summary_df["paired_delta_ci_low"] = [0.0, delta_low]
    summary_df["paired_delta_ci_high"] = [0.0, delta_high]
    summary_df["accept"] = [False, bool(delta_low > 0.0)]
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/dog_ab_test_phase_a_details.csv", index=False)
    summary_df.to_csv("results/dog_ab_test_phase_a.csv", index=False)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
