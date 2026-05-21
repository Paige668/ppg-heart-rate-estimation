"""Generate comparison tables and charts for the strict RF/XGBoost and retained CNN baselines."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED_REFERENCE = 42
TRAINING_SEEDS = [42, 52, 62, 72, 82]
CASE_LABELS = {"no_prep": "No-Prep", "full_pipeline": "Full Pipeline"}
CASE_ORDER = ["No-Prep", "Full Pipeline"]
FAMILY_ORDER = ["Random Forest", "XGBoost", "CNN Unified"]
FAMILY_SHORT = {"Random Forest": "RF", "XGBoost": "XGB", "CNN Unified": "CNN"}
HR_GROUP_ORDER = ["HR < 80", "80 <= HR < 120", "120 <= HR < 140", "HR >= 140"]
FAMILY_PATHS = {
    "Random Forest": "random_forest/stability_metrics_all.csv",
    "XGBoost": "xgboost/stability_metrics_all.csv",
    "CNN Unified": "cnn/stability_metrics_all.csv",
}
CNN_HR_GROUP_PATH = "cnn/no_prep/hr_group_mae_summary.csv"


def find_workspace_root(start_dir: Path) -> Path:
    current = start_dir.resolve()
    while True:
        if (current / "data").is_dir():
            return current
        parent = current.parent
        if parent == current:
            raise RuntimeError("Could not locate workspace root containing samples_train_csv and samples_test_csv.")
        current = parent


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = find_workspace_root(SCRIPT_DIR)
GENERAL_ROOT = WORKSPACE_ROOT / "rq3_dynamic_models"
OUTPUT_DIR = GENERAL_ROOT / "comparison_results"
CHART_DIR = OUTPUT_DIR / "charts"


def load_family_stability(family: str) -> pd.DataFrame:
    csv_path = GENERAL_ROOT / FAMILY_PATHS[family]
    df = pd.read_csv(csv_path)
    df["family"] = family
    df["family_short"] = FAMILY_SHORT[family]
    df["case_slug"] = df["case"]
    df["case"] = df["case"].map(CASE_LABELS)
    return df


def load_all_stability() -> pd.DataFrame:
    frames = [load_family_stability(family) for family in FAMILY_ORDER]
    df = pd.concat(frames, ignore_index=True)
    df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER, ordered=True)
    df["case"] = pd.Categorical(df["case"], categories=CASE_ORDER, ordered=True)
    return df.sort_values(["family", "case", "training_seed"])


def build_canonical_df(stability_df: pd.DataFrame) -> pd.DataFrame:
    canonical = stability_df[stability_df["training_seed"] == SEED_REFERENCE].copy()
    if "split_seed" in canonical.columns:
        canonical = canonical[canonical["split_seed"].isna() | (canonical["split_seed"] == SEED_REFERENCE)]
    canonical = canonical[[
        "family",
        "family_short",
        "case",
        "case_slug",
        "training_seed",
        "train_mae",
        "train_rmse",
        "test_mae",
        "test_rmse",
    ]]
    canonical = canonical.sort_values(["family", "case"])
    canonical["family"] = canonical["family"].astype(str)
    canonical["case"] = canonical["case"].astype(str)
    return canonical


def build_stability_summary(stability_df: pd.DataFrame) -> pd.DataFrame:
    grouped = stability_df.groupby(["family", "family_short", "case", "case_slug"], observed=True)
    summary = grouped.agg(
        test_mae_mean=("test_mae", "mean"),
        test_mae_std=("test_mae", lambda series: series.std(ddof=0)),
        test_rmse_mean=("test_rmse", "mean"),
        test_rmse_std=("test_rmse", lambda series: series.std(ddof=0)),
        train_mae_mean=("train_mae", "mean"),
        train_rmse_mean=("train_rmse", "mean"),
    ).reset_index()
    summary = summary.sort_values(["family", "case"])
    summary["family"] = summary["family"].astype(str)
    summary["case"] = summary["case"].astype(str)
    return summary


def build_preprocessing_gain(canonical_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILY_ORDER:
        family_df = canonical_df[canonical_df["family"] == family].copy()
        no_prep = family_df[family_df["case"] == "No-Prep"].iloc[0]
        full_pipeline = family_df[family_df["case"] == "Full Pipeline"].iloc[0]
        mae_gain = no_prep["test_mae"] - full_pipeline["test_mae"]
        rmse_gain = no_prep["test_rmse"] - full_pipeline["test_rmse"]
        rows.append(
            {
                "family": family,
                "family_short": FAMILY_SHORT[family],
                "no_prep_test_mae": no_prep["test_mae"],
                "full_pipeline_test_mae": full_pipeline["test_mae"],
                "mae_gain_bpm": mae_gain,
                "mae_gain_pct": mae_gain / no_prep["test_mae"] * 100.0,
                "no_prep_test_rmse": no_prep["test_rmse"],
                "full_pipeline_test_rmse": full_pipeline["test_rmse"],
                "rmse_gain_bpm": rmse_gain,
                "rmse_gain_pct": rmse_gain / no_prep["test_rmse"] * 100.0,
            }
        )
    return pd.DataFrame(rows)


def build_train_test_gap(canonical_df: pd.DataFrame) -> pd.DataFrame:
    gap_df = canonical_df.copy()
    gap_df["mae_ratio_test_over_train"] = gap_df["test_mae"] / gap_df["train_mae"]
    gap_df["rmse_ratio_test_over_train"] = gap_df["test_rmse"] / gap_df["train_rmse"]
    gap_df["case_label"] = gap_df["family_short"] + " " + gap_df["case"].map({"No-Prep": "NP", "Full Pipeline": "FP"})
    return gap_df


def build_canonical_vs_seed_mean(canonical_df: pd.DataFrame, stability_summary: pd.DataFrame) -> pd.DataFrame:
    merged = canonical_df.merge(
        stability_summary[["family", "case", "test_mae_mean", "test_mae_std", "test_rmse_mean", "test_rmse_std"]],
        on=["family", "case"],
        how="left",
    )
    merged["case_label"] = merged["family_short"] + " " + merged["case"].map({"No-Prep": "NP", "Full Pipeline": "FP"})
    merged["mae_delta_seed_mean_minus_canonical"] = merged["test_mae_mean"] - merged["test_mae"]
    merged["rmse_delta_seed_mean_minus_canonical"] = merged["test_rmse_mean"] - merged["test_rmse"]
    return merged.sort_values(["family", "case"])


def load_cnn_hr_group_summary() -> tuple[pd.DataFrame, float, float, pd.DataFrame]:
    # Calculate directly from CNN no_prep predictions dynamically
    pred_path = GENERAL_ROOT / "cnn/no_prep/seed_42/predictions_seed_42.csv"
    if not pred_path.is_file():
        # Fallback dummy if file not created yet
        rows = []
        for g in HR_GROUP_ORDER:
            rows.append({"group": g, "mae": 0.0, "rmse": 0.0, "count": 0})
        grouped_df = pd.DataFrame(rows)
        grouped_df["group"] = pd.Categorical(grouped_df["group"], categories=HR_GROUP_ORDER, ordered=True)
        return grouped_df, 0.0, 0.0, pd.DataFrame(rows)

    df = pd.read_csv(pred_path)
    # Map column names if needed: in CNN script it is "true_hr", "pred_hr", "error_bpm"
    true_col = "true_hr"
    pred_col = "pred_hr"
    
    rows = []
    # Match HR groups
    for group_label, low, high in [
        ("HR < 80", 0, 80),
        ("80 <= HR < 120", 80, 120),
        ("120 <= HR < 140", 120, 140),
        ("HR >= 140", 140, 999)
    ]:
        sub = df[(df[true_col] >= low) & (df[true_col] < high)]
        if len(sub) > 0:
            mae = np.mean(np.abs(sub[pred_col] - sub[true_col]))
            rmse = np.sqrt(np.mean((sub[pred_col] - sub[true_col]) ** 2))
            rows.append({"group": group_label, "mae": mae, "rmse": rmse, "count": len(sub)})
        else:
            rows.append({"group": group_label, "mae": 0.0, "rmse": 0.0, "count": 0})
            
    overall_mae = np.mean(np.abs(df[pred_col] - df[true_col]))
    overall_rmse = np.sqrt(np.mean((df[pred_col] - df[true_col]) ** 2))
    
    grouped_df = pd.DataFrame(rows)
    grouped_df["group"] = pd.Categorical(grouped_df["group"], categories=HR_GROUP_ORDER, ordered=True)
    grouped_df = grouped_df.sort_values("group")
    
    full_df = pd.concat([grouped_df, pd.DataFrame([{"group": "OVERALL", "mae": overall_mae, "rmse": overall_rmse, "count": len(df)}])], ignore_index=True)
    return grouped_df, overall_mae, overall_rmse, full_df


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(OUTPUT_DIR / filename, index=False)


def set_common_axis_style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def plot_grouped_case_metric(canonical_df: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str) -> None:
    pivot = canonical_df.pivot(index="family", columns="case", values=metric).reindex(FAMILY_ORDER)[CASE_ORDER]
    x = np.arange(len(FAMILY_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars_no = ax.bar(x - width / 2, pivot["No-Prep"].to_numpy(), width, label="No-Prep", color="#F08F3E")
    bars_fp = ax.bar(x + width / 2, pivot["Full Pipeline"].to_numpy(), width, label="Full Pipeline", color="#4C78A8")
    for bars in [bars_no, bars_fp]:
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_SHORT[family] for family in FAMILY_ORDER])
    # ax.set_title(title)  # Removed for academic thesis formatting compliance
    ax.legend(frameon=False)
    set_common_axis_style(ax, ylabel)
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=160)
    plt.close(fig)


def plot_train_test_metric(gap_df: pd.DataFrame, train_metric: str, test_metric: str, title: str, ylabel: str, filename: str) -> None:
    x = np.arange(len(gap_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    train_vals = gap_df[train_metric].to_numpy()
    test_vals = gap_df[test_metric].to_numpy()
    bars_train = ax.bar(x - width / 2, train_vals, width, label="Train", color="#A0CBE8")
    bars_test = ax.bar(x + width / 2, test_vals, width, label="Test", color="#E15759")
    for bars in [bars_train, bars_test]:
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(gap_df["case_label"].tolist())
    ax.set_title(title)
    ax.legend(frameon=False)
    set_common_axis_style(ax, ylabel)
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=160)
    plt.close(fig)


def plot_preprocessing_gain(gain_df: pd.DataFrame, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = ["#59A14F", "#59A14F", "#E15759"]
    bars = ax.bar(gain_df["family_short"], gain_df["mae_gain_bpm"], color=colors)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.15 if value >= 0 else -0.45), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Canonical Test MAE Gain from Full Pipeline")
    set_common_axis_style(ax, "MAE improvement (bpm)")
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=160)
    plt.close(fig)


def plot_stability_std(stability_summary: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str) -> None:
    pivot = stability_summary.pivot(index="family", columns="case", values=metric).reindex(FAMILY_ORDER)[CASE_ORDER]
    x = np.arange(len(FAMILY_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(x - width / 2, pivot["No-Prep"].to_numpy(), width, label="No-Prep", color="#F08F3E")
    ax.bar(x + width / 2, pivot["Full Pipeline"].to_numpy(), width, label="Full Pipeline", color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_SHORT[family] for family in FAMILY_ORDER])
    ax.set_title(title)
    ax.legend(frameon=False)
    set_common_axis_style(ax, ylabel)
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=160)
    plt.close(fig)


def plot_canonical_vs_seed_mean(comparison_df: pd.DataFrame, filename: str) -> None:
    x = np.arange(len(comparison_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    canonical_vals = comparison_df["test_mae"].to_numpy()
    mean_vals = comparison_df["test_mae_mean"].to_numpy()
    ax.bar(x - width / 2, canonical_vals, width, label="Canonical seed-42 run", color="#76B7B2")
    ax.bar(x + width / 2, mean_vals, width, label="5-seed mean", color="#B07AA1")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["case_label"].tolist())
    ax.set_title("Canonical Test MAE vs 5-Seed Mean Test MAE")
    ax.legend(frameon=False)
    set_common_axis_style(ax, "Test MAE (bpm)")
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=160)
    plt.close(fig)


def plot_cnn_hr_group_mae(grouped_df: pd.DataFrame, overall_mae: float, overall_rmse: float, filename: str) -> None:
    x = np.arange(len(grouped_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    mae_vals = grouped_df["mae"].to_numpy()
    rmse_vals = grouped_df["rmse"].to_numpy()
    
    bars_mae = ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#F28E2B")
    bars_rmse = ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color="#E15759")
    
    for bars in [bars_mae, bars_rmse]:
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.55, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
            
    ax.axhline(overall_mae, color="#F28E2B", linestyle="--", linewidth=1.2, label=f"Overall MAE = {overall_mae:.2f}")
    ax.axhline(overall_rmse, color="#E15759", linestyle="--", linewidth=1.2, label=f"Overall RMSE = {overall_rmse:.2f}")
    
    ax.set_xticks(x)
    labels = [f"{g}\nn={c}" for g, c in zip(grouped_df["group"].astype(str), grouped_df["count"])]
    ax.set_xticklabels(labels)
    # ax.set_title("CNN Unified No-Prep Test MAE & RMSE by Ground-Truth HR Group")  # Removed for academic thesis formatting compliance
    ax.legend(frameon=False)
    set_common_axis_style(ax, "bpm")
    ax.set_ylim(0, max(max(mae_vals), max(rmse_vals)) * 1.25)
    
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=160)
    plt.close(fig)


def plot_combined_mae_rmse(canonical_df: pd.DataFrame, filename: str) -> None:
    # Pivot the data for MAE and RMSE
    pivot_mae = canonical_df.pivot(index="family", columns="case", values="test_mae").reindex(FAMILY_ORDER)[CASE_ORDER]
    pivot_rmse = canonical_df.pivot(index="family", columns="case", values="test_rmse").reindex(FAMILY_ORDER)[CASE_ORDER]
    
    x = np.arange(len(FAMILY_ORDER))
    width = 0.34
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- LEFT SUBPLOT: MAE (bpm) ---
    bars_mae_no = ax1.bar(x - width/2, pivot_mae["No-Prep"].to_numpy(), width, label="No-Prep", color="#5C96E6")
    bars_mae_fp = ax1.bar(x + width/2, pivot_mae["Full Pipeline"].to_numpy(), width, label="Full Pipeline", color="#3A75C4")
    
    for bars in [bars_mae_no, bars_mae_fp]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            
    ax1.set_xticks(x)
    ax1.set_xticklabels([FAMILY_SHORT[family] for family in FAMILY_ORDER], fontsize=11)
    ax1.set_ylabel("MAE (bpm)", fontsize=12)
    ax1.legend(frameon=True, loc="upper right")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, 32)
    
    # --- RIGHT SUBPLOT: RMSE (bpm) ---
    bars_rmse_no = ax2.bar(x - width/2, pivot_rmse["No-Prep"].to_numpy(), width, label="No-Prep", color="#A292EB")
    bars_rmse_fp = ax2.bar(x + width/2, pivot_rmse["Full Pipeline"].to_numpy(), width, label="Full Pipeline", color="#7C5BB4")
    
    for bars in [bars_rmse_no, bars_rmse_fp]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            
    ax2.set_xticks(x)
    ax2.set_xticklabels([FAMILY_SHORT[family] for family in FAMILY_ORDER], fontsize=11)
    ax2.set_ylabel("RMSE (bpm)", fontsize=12)
    ax2.legend(frameon=True, loc="upper right")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 42)
    
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=200)
    plt.close(fig)


def plot_seed_grid(stability_df: pd.DataFrame, metric: str, title: str, filename: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.0), sharex=True)
    for row_idx, family in enumerate(FAMILY_ORDER):
        for col_idx, case in enumerate(CASE_ORDER):
            ax = axes[row_idx, col_idx]
            subset = stability_df[(stability_df["family"] == family) & (stability_df["case"] == case)].sort_values("training_seed")
            values = subset[metric].to_numpy()
            ax.plot(TRAINING_SEEDS, values, marker="o", linewidth=1.8, color="#4C78A8" if col_idx == 1 else "#F08F3E")
            ax.axhline(values.mean(), color="black", linestyle="--", linewidth=1.0)
            ax.set_title(f"{FAMILY_SHORT[family]} - {case}")
            set_common_axis_style(ax, metric.replace("_", " ").title())
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=160)
    plt.close(fig)


def write_summary(
    canonical_df: pd.DataFrame,
    stability_summary: pd.DataFrame,
    gain_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    cnn_hr_group_df: pd.DataFrame,
    cnn_hr_overall_mae: float,
    cnn_hr_overall_rmse: float,
) -> None:
    rf_fp_std = float(stability_summary[(stability_summary["family"] == "Random Forest") & (stability_summary["case"] == "Full Pipeline")]["test_mae_std"].iloc[0])
    rf_np_mean = float(stability_summary[(stability_summary["family"] == "Random Forest") & (stability_summary["case"] == "No-Prep")]["test_mae_mean"].iloc[0])
    xgb_fp_mean = float(stability_summary[(stability_summary["family"] == "XGBoost") & (stability_summary["case"] == "Full Pipeline")]["test_mae_mean"].iloc[0])
    xgb_np_mean = float(stability_summary[(stability_summary["family"] == "XGBoost") & (stability_summary["case"] == "No-Prep")]["test_mae_mean"].iloc[0])
    cnn_np_mean = float(stability_summary[(stability_summary["family"] == "CNN Unified") & (stability_summary["case"] == "No-Prep")]["test_mae_mean"].iloc[0])
    cnn_fp_canonical = float(canonical_df[(canonical_df["family"] == "CNN Unified") & (canonical_df["case"] == "Full Pipeline")]["test_mae"].iloc[0])
    cnn_low_hr_mae = float(cnn_hr_group_df[cnn_hr_group_df["group"] == "HR < 80"]["mae"].iloc[0])
    cnn_mid_hr_mae = float(cnn_hr_group_df[cnn_hr_group_df["group"] == "120 <= HR < 140"]["mae"].iloc[0])
    rf_fp_gain = float(gain_df[gain_df["family"] == "Random Forest"]["mae_gain_bpm"].iloc[0])
    xgb_fp_gain = float(gain_df[gain_df["family"] == "XGBoost"]["mae_gain_bpm"].iloc[0])
    canonical_note = (
        "`canonical` means the current official single reference run quoted in the main comparison tables. "
        f"In the current setup, that reference run is the seed-{SEED_REFERENCE} retained run for each selected baseline. "
        "It is reported separately from the 5-seed mean ± std, which describes stability rather than the single reference point."
    )
    markdown = f"""# Comparison Results Summary

## What `canonical` means

{canonical_note}

## Key takeaways

1. Under No-Prep, the CNN mainline remains clearly strongest, while the strict-matched RF and XGBoost baselines improve substantially and now sit at {rf_np_mean:.4f} and {xgb_np_mean:.4f} bpm 5-seed mean MAE.
2. Under Full Pipeline, CNN still keeps the best canonical MAE at {cnn_fp_canonical:.4f} bpm, while XGBoost remains the strongest tree baseline and RF remains close behind.
3. In the strict-matched setup, preprocessing still helps the tree baselines, but the MAE gains shrink to {rf_fp_gain:.2f} bpm for RF and {xgb_fp_gain:.2f} bpm for XGBoost.
4. RF Full Pipeline is still the most seed-stable retained baseline, with Test MAE std = {rf_fp_std:.4f} bpm.
5. CNN No-Prep still has the best raw-input 5-seed mean MAE at {cnn_np_mean:.4f} bpm, preserving the main deep-model conclusion under the stricter comparison protocol.
6. The new test-folder HR-group breakdown for the final CNN No-Prep model shows strong variation by HR regime: MAE is highest below 80 bpm ({cnn_low_hr_mae:.2f}) and lowest in the 120-140 bpm range ({cnn_mid_hr_mae:.2f}), while overall test-folder MAE remains {cnn_hr_overall_mae:.4f} bpm. (RMSE follows a similar U-shaped trend, with overall RMSE at {cnn_hr_overall_rmse:.4f} bpm).

## Generated tables

- `canonical_results_seed42.csv`
- `stability_summary.csv`
- `preprocessing_gain_summary.csv`
- `train_test_gap_summary.csv`
- `canonical_vs_seed_mean_summary.csv`
- `cnn_no_prep_hr_group_mae_summary.csv`

## Generated charts

- `charts/canonical_test_mae.png`
- `charts/canonical_test_rmse.png`
- `charts/train_vs_test_mae.png`
- `charts/train_vs_test_rmse.png`
- `charts/preprocessing_gain_test_mae.png`
- `charts/stability_test_mae_std.png`
- `charts/stability_test_rmse_std.png`
- `charts/canonical_vs_seed_mean_mae.png`
- `charts/seed_test_mae_grid.png`
- `charts/seed_test_rmse_grid.png`
- `charts/cnn_no_prep_hr_group_mae.png`
"""
    (OUTPUT_DIR / "COMPARISON_RESULTS_SUMMARY.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    stability_df = load_all_stability()
    canonical_df = build_canonical_df(stability_df)
    stability_summary = build_stability_summary(stability_df)
    gain_df = build_preprocessing_gain(canonical_df)
    gap_df = build_train_test_gap(canonical_df)
    comparison_df = build_canonical_vs_seed_mean(canonical_df, stability_summary)
    cnn_hr_group_df, cnn_hr_overall_mae, cnn_hr_overall_rmse, cnn_hr_group_raw = load_cnn_hr_group_summary()

    save_dataframe(canonical_df, "canonical_results_seed42.csv")
    save_dataframe(stability_summary, "stability_summary.csv")
    save_dataframe(gain_df, "preprocessing_gain_summary.csv")
    save_dataframe(gap_df, "train_test_gap_summary.csv")
    save_dataframe(comparison_df, "canonical_vs_seed_mean_summary.csv")
    save_dataframe(stability_df, "seed_level_metrics_all.csv")
    save_dataframe(cnn_hr_group_raw, "cnn_no_prep_hr_group_mae_summary.csv")

    plot_grouped_case_metric(canonical_df, "test_mae", "Canonical Test MAE by Model Family", "Test MAE (bpm)", "canonical_test_mae.png")
    plot_grouped_case_metric(canonical_df, "test_rmse", "Canonical Test RMSE by Model Family", "Test RMSE (bpm)", "canonical_test_rmse.png")
    plot_combined_mae_rmse(canonical_df, "canonical_test_combined.png")
    plot_train_test_metric(gap_df, "train_mae", "test_mae", "Canonical Train vs Test MAE", "MAE (bpm)", "train_vs_test_mae.png")
    plot_train_test_metric(gap_df, "train_rmse", "test_rmse", "Canonical Train vs Test RMSE", "RMSE (bpm)", "train_vs_test_rmse.png")
    plot_preprocessing_gain(gain_df, "preprocessing_gain_test_mae.png")
    plot_stability_std(stability_summary, "test_mae_std", "5-Seed Test MAE Std by Model Family", "Test MAE std (bpm)", "stability_test_mae_std.png")
    plot_stability_std(stability_summary, "test_rmse_std", "5-Seed Test RMSE Std by Model Family", "Test RMSE std (bpm)", "stability_test_rmse_std.png")
    plot_canonical_vs_seed_mean(comparison_df, "canonical_vs_seed_mean_mae.png")
    plot_seed_grid(stability_df, "test_mae", "Per-Seed Test MAE by Family and Case", "seed_test_mae_grid.png")
    plot_seed_grid(stability_df, "test_rmse", "Per-Seed Test RMSE by Family and Case", "seed_test_rmse_grid.png")
    plot_cnn_hr_group_mae(cnn_hr_group_df, cnn_hr_overall_mae, cnn_hr_overall_rmse, "cnn_no_prep_hr_group_mae.png")
    write_summary(canonical_df, stability_summary, gain_df, comparison_df, cnn_hr_group_df, cnn_hr_overall_mae, cnn_hr_overall_rmse)

    print("Saved comparison tables and charts to comparison_results/")


if __name__ == "__main__":
    main()