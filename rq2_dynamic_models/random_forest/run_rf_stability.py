"""
Run 5-seed stability analysis for the strict-matched Random Forest baseline.

This keeps the strict matched feature pipeline fixed and varies only the RF seed.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from run_rf_matched import run_rf_experiment


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_SEEDS = [42, 52, 62, 72, 82]
CASES = [
    ("no_prep", False),
    ("full_pipeline", True),
]


def run_case(case_slug, use_full_pipeline):
    case_dir = os.path.join(CURRENT_DIR, case_slug)
    os.makedirs(case_dir, exist_ok=True)

    rows = []
    for training_seed in TRAINING_SEEDS:
        seed_dir = os.path.join(case_dir, f"seed_{training_seed}")
        result = run_rf_experiment(
            use_full_pipeline=use_full_pipeline,
            case_name=case_slug,
            training_seed=training_seed,
            output_dir=seed_dir,
            prediction_file_name=f"rf_strict_{case_slug}_seed_{training_seed}.csv",
        )
        rows.append(result)

    case_df = pd.DataFrame(rows).sort_values("training_seed")
    case_df.to_csv(os.path.join(case_dir, "seed_metrics.csv"), index=False)
    return case_df


def plot_case(case_df, case_slug, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    metrics = [
        ("mae", "Test MAE (bpm)", "#4C78A8"),
        ("rmse", "Test RMSE (bpm)", "#F58518"),
    ]

    for ax, (column, ylabel, color) in zip(axes, metrics):
        x = range(len(case_df))
        values = case_df[column].to_numpy()
        labels = [str(seed) for seed in case_df["training_seed"]]
        bars = ax.bar(x, values, color=color, edgecolor="white")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        mean_val = values.mean()
        std_val = values.std(ddof=0)
        ax.axhline(mean_val, color="black", linestyle="--", linewidth=1.2, label=f"mean={mean_val:.3f}")
        ax.fill_between(
            [-0.5, len(values) - 0.5],
            mean_val - std_val,
            mean_val + std_val,
            color="gray",
            alpha=0.15,
            label=f"std={std_val:.3f}",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Model seed")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, f"{case_slug}_stability.png"), dpi=150)
    plt.close(fig)


def summarize(df):
    return {
        "mae_mean": df["mae"].mean(),
        "mae_std": df["mae"].std(ddof=0),
        "rmse_mean": df["rmse"].mean(),
        "rmse_std": df["rmse"].std(ddof=0),
    }


def write_summary(no_prep_df, full_pipeline_df):
    all_df = pd.concat([no_prep_df, full_pipeline_df], ignore_index=True)
    all_df.to_csv(os.path.join(CURRENT_DIR, "stability_metrics_all.csv"), index=False)

    no_stats = summarize(no_prep_df)
    fp_stats = summarize(full_pipeline_df)

    markdown = f"""# Random Forest Strict Matched Stability Analysis

Model seeds used: {', '.join(str(seed) for seed in TRAINING_SEEDS)}

| Case | Test MAE mean ± std | Test RMSE mean ± std |
| ---- | ---- | ---- |
| No-Prep | {no_stats['mae_mean']:.4f} ± {no_stats['mae_std']:.4f} | {no_stats['rmse_mean']:.4f} ± {no_stats['rmse_std']:.4f} |
| Full Pipeline | {fp_stats['mae_mean']:.4f} ± {fp_stats['mae_std']:.4f} | {fp_stats['rmse_mean']:.4f} ± {fp_stats['rmse_std']:.4f} |
"""

    with open(os.path.join(CURRENT_DIR, "stability_analysis.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown)


def main():
    no_prep_df = run_case("no_prep", False)
    full_pipeline_df = run_case("full_pipeline", True)
    plot_case(no_prep_df, "no_prep", "Random Forest Strict Matched Stability - No Prep")
    plot_case(full_pipeline_df, "full_pipeline", "Random Forest Strict Matched Stability - Full Pipeline")
    write_summary(no_prep_df, full_pipeline_df)


if __name__ == "__main__":
    main()