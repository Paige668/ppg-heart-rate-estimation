"""
Run 5-seed stability analysis for the retained general unified CNN baseline.

This analysis keeps the train/test split and the train/validation split fixed,
while varying only the training seed.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTE_DIR = os.path.dirname(CURRENT_DIR)
if ROUTE_DIR not in sys.path:
    sys.path.insert(0, ROUTE_DIR)

from cnn_unified import build_unified_config, run_experiment


SPLIT_SEED = 42
TRAINING_SEEDS = [42, 52, 62, 72, 82]


def run_case(case_slug, use_full_pipeline):
    case_dir = os.path.join(CURRENT_DIR, case_slug)
    os.makedirs(case_dir, exist_ok=True)

    rows = []
    for training_seed in TRAINING_SEEDS:
        seed_dir = os.path.join(case_dir, f"seed_{training_seed}")
        model_name = (
            f"CNN Unified General Baseline ({'Full Pipeline' if use_full_pipeline else 'No Prep'}) "
            f"- Stability Seed {training_seed}"
        )
        metrics = run_experiment(
            config=build_unified_config(model_name=model_name, use_full_pipeline=use_full_pipeline),
            out_dir=seed_dir,
            prediction_file_name=f"predictions_seed_{training_seed}.csv",
            training_seed=training_seed,
            split_seed=SPLIT_SEED,
        )
        rows.append({"case": case_slug, **metrics})

    case_df = pd.DataFrame(rows).sort_values("training_seed")
    case_df.to_csv(os.path.join(case_dir, "seed_metrics.csv"), index=False)
    return case_df


def plot_case(case_df, case_slug, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    metrics = [("test_mae", "Test MAE (bpm)", "#4C78A8"), ("test_rmse", "Test RMSE (bpm)", "#F58518")]

    for ax, (column, ylabel, color) in zip(axes, metrics):
        x = np.arange(len(case_df))
        values = case_df[column].to_numpy()
        labels = [str(seed) for seed in case_df["training_seed"]]
        bars = ax.bar(x, values, color=color, edgecolor="white")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        mean_val = values.mean()
        std_val = values.std(ddof=0)
        ax.axhline(mean_val, color="black", linestyle="--", linewidth=1.2, label=f"mean={mean_val:.3f}")
        ax.fill_between([-0.5, len(values) - 0.5], mean_val - std_val, mean_val + std_val, color="gray", alpha=0.15, label=f"std={std_val:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Training seed")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, f"{case_slug}_stability.png"), dpi=150)
    plt.close(fig)


def write_summary(no_prep_df, full_pipeline_df):
    all_df = pd.concat([no_prep_df, full_pipeline_df], ignore_index=True)
    all_df.to_csv(os.path.join(CURRENT_DIR, "stability_metrics_all.csv"), index=False)

    def summarize(df):
        return {
            "mae_mean": df["test_mae"].mean(),
            "mae_std": df["test_mae"].std(ddof=0),
            "rmse_mean": df["test_rmse"].mean(),
            "rmse_std": df["test_rmse"].std(ddof=0),
        }

    no_stats = summarize(no_prep_df)
    fp_stats = summarize(full_pipeline_df)

    markdown = f"""# General CNN Baseline Stability Analysis

This analysis evaluates the retained general unified CNN baseline under multiple training seeds while keeping:

- the train/test split fixed
- the train/validation split fixed (`split_seed={SPLIT_SEED}`)
- the model architecture, feature construction, and loss definition unchanged

Only the training randomness changes across runs: initialization, shuffling, and augmentation randomness.

## Seeds

Training seeds used: {', '.join(str(seed) for seed in TRAINING_SEEDS)}

## Summary

| Case | Test MAE mean ± std | Test RMSE mean ± std |
| ---- | ---- | ---- |
| No-Prep | {no_stats['mae_mean']:.4f} ± {no_stats['mae_std']:.4f} | {no_stats['rmse_mean']:.4f} ± {no_stats['rmse_std']:.4f} |
| Full Pipeline | {fp_stats['mae_mean']:.4f} ± {fp_stats['mae_std']:.4f} | {fp_stats['rmse_mean']:.4f} ± {fp_stats['rmse_std']:.4f} |

## Interpretation

1. This stability analysis uses the same retained general CNN baseline in both cases, so the only case-level difference remains whether preprocessing is applied.
2. The seed sweep tests whether the retained baseline result is stable or whether it depends too heavily on one favorable initialization.
3. The reported `mean ± std` values are suitable for documenting reproducibility of the final selected baseline.
"""

    with open(os.path.join(CURRENT_DIR, "stability_analysis.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown)


def main():
    no_prep_df = run_case("no_prep", use_full_pipeline=False)
    full_pipeline_df = run_case("full_pipeline", use_full_pipeline=True)
    plot_case(no_prep_df, "no_prep", "General CNN Baseline Stability - No Prep")
    plot_case(full_pipeline_df, "full_pipeline", "General CNN Baseline Stability - Full Pipeline")
    write_summary(no_prep_df, full_pipeline_df)

    print("Saved: stability_metrics_all.csv")
    print("Saved: no_prep/seed_metrics.csv")
    print("Saved: full_pipeline/seed_metrics.csv")
    print("Saved: no_prep_stability.png")
    print("Saved: full_pipeline_stability.png")
    print("Saved: stability_analysis.md")


if __name__ == "__main__":
    main()