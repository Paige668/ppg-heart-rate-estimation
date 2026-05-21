"""
rq1_cross_validation.py — 5-Fold Cross-Validation for RQ1 (Traditional ML Models)

Runs 5-fold cross-validation on the resting-state PPG feature dataset for:
  - Linear Regression
  - K-Nearest Neighbors (KNN, K=10)
  - Random Forest

Outputs:
  - results/rq1_cv_results.csv       — per-fold metrics
  - results/rq1_cv_summary.csv       — mean ± std summary table
  - figures/rq1_cv_mae_comparison.png
  - figures/rq1_cv_rmse_comparison.png
  - figures/rq1_cv_combined.png      — combined figure for thesis
"""

import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(os.path.dirname(BASE_DIR), "data", "ppg_10s_windows_rest")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR    = os.path.join(BASE_DIR, "figures")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

FS       = 25.0
N_FOLDS  = 5
RAND     = 42

# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction  (reused from rf_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_peaks(sig, fs):
    min_dist   = int(fs * 60 / 100)
    prominence = sig.std() * 0.3
    peaks, _   = signal.find_peaks(sig, distance=min_dist, prominence=prominence)
    return peaks


def extract_features(sig, fs=FS):
    feats = {}
    feats["mean"]     = np.mean(sig)
    feats["std"]      = np.std(sig)
    feats["skewness"] = float(stats.skew(sig))
    feats["kurtosis"] = float(stats.kurtosis(sig))
    feats["min"]      = np.min(sig)
    feats["max"]      = np.max(sig)
    feats["mad"]      = np.mean(np.abs(sig - np.mean(sig)))

    feats["peak_to_peak"] = feats["max"] - feats["min"]
    peaks = _detect_peaks(sig, fs)
    feats["n_peaks"] = len(peaks)
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        feats["rr_mean"] = np.mean(rr)
        feats["rr_std"]  = np.std(rr)
        rr_diff = np.diff(rr)
        feats["rmssd"] = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0.0
    else:
        feats["rr_mean"] = np.nan
        feats["rr_std"]  = np.nan
        feats["rmssd"]   = np.nan

    n     = len(sig)
    freqs = fftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(fft(sig))[:n // 2]
    freqs   = freqs[:n // 2]
    band_mask  = (freqs >= 0.5) & (freqs <= 4.0)
    band_power = np.sum(fft_mag[band_mask] ** 2)
    feats["band_power"] = band_power
    if band_mask.any():
        dominant_idx = np.argmax(fft_mag[band_mask])
        feats["dominant_freq"]      = freqs[band_mask][dominant_idx]
        feats["dominant_amplitude"] = fft_mag[band_mask][dominant_idx]
    else:
        feats["dominant_freq"]      = np.nan
        feats["dominant_amplitude"] = np.nan
    psd      = fft_mag ** 2
    psd_norm = psd / (psd.sum() + 1e-12)
    feats["spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    feats["energy"] = np.sum(sig ** 2) / n
    zero_crossings  = np.diff(np.sign(sig - np.mean(sig)))
    feats["zcr"]    = np.sum(zero_crossings != 0) / n

    dsig = np.diff(sig)
    feats["hjorth_mobility"] = np.sqrt(np.var(dsig) / (np.var(sig) + 1e-12))
    troughs, _ = signal.find_peaks(-sig, distance=int(fs * 60 / 100),
                                   prominence=sig.std() * 0.3)
    if len(peaks) >= 1 and len(troughs) >= 1:
        fall_times = []
        for p in peaks:
            after = troughs[troughs > p]
            if len(after):
                fall_times.append((after[0] - p) / fs)
        feats["fall_time"] = np.mean(fall_times) if fall_times else np.nan
    else:
        feats["fall_time"] = np.nan

    return feats


def load_dataset(folder, ppg_col="PPG_filtered", hr_col="HR"):
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    print(f"  Loading {len(csv_files)} CSV files ...")
    records   = []
    hr_values = []
    for i, f in enumerate(csv_files):
        df  = pd.read_csv(f)
        sig = df[ppg_col].values.astype(float)
        hr  = float(df[hr_col].iloc[0])
        records.append(extract_features(sig, fs=FS))
        hr_values.append(hr)
        if (i + 1) % 300 == 0:
            print(f"    {i + 1}/{len(csv_files)} processed ...")
    feat_df = pd.DataFrame(records)
    feat_df = feat_df.fillna(feat_df.median())
    return feat_df.values, np.array(hr_values), list(feat_df.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(X, y, models, n_folds=N_FOLDS, seed=RAND):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    results = {name: {"mae": [], "rmse": []} for name in models}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n  ── Fold {fold_idx + 1}/{n_folds} ──")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        for name, model in models.items():
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_val_s)
            mae  = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            results[name]["mae"].append(mae)
            results[name]["rmse"].append(rmse)
            print(f"    [{name:20s}]  MAE={mae:.3f}  RMSE={rmse:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Linear Regression": "#4C72B0",
    "KNN (K=10)":        "#DD8452",
    "Random Forest":     "#55A868",
}

def plot_metric(summary_df, metric_col, err_col, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    models  = summary_df["Model"].tolist()
    vals    = summary_df[metric_col].tolist()
    errs    = summary_df[err_col].tolist()
    colors  = [COLORS.get(m, "#888888") for m in models]

    bars = ax.bar(models, vals, yerr=errs, capsize=7,
                  color=colors, edgecolor="white", linewidth=1.2,
                  error_kw={"elinewidth": 2, "ecolor": "#333333"}, zorder=3)

    # Value labels
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + e + 0.05,
                f"{v:.2f}±{e:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_combined(summary_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "5-Fold Cross-Validation Results — Resting-State PPG Feature Models (RQ1)",
        fontsize=13, fontweight="bold", y=1.02
    )

    metrics = [
        ("MAE_mean", "MAE_std",  "MAE [bpm]",  "Mean Absolute Error (MAE)"),
        ("RMSE_mean", "RMSE_std", "RMSE [bpm]", "Root Mean Square Error (RMSE)"),
    ]

    models = summary_df["Model"].tolist()
    colors = [COLORS.get(m, "#888888") for m in models]

    for ax, (mean_col, std_col, ylabel, subtitle) in zip(axes, metrics):
        vals = summary_df[mean_col].tolist()
        errs = summary_df[std_col].tolist()

        bars = ax.bar(models, vals, yerr=errs, capsize=7,
                      color=colors, edgecolor="white", linewidth=1.2,
                      error_kw={"elinewidth": 2, "ecolor": "#333333"}, zorder=3)

        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + e + 0.05,
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(subtitle, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylim(0, max(vals) * 1.4)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=10)

    # Shared legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in models]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RQ1 — 5-Fold Cross-Validation (Traditional ML Models)")
    print("=" * 60)

    # Load data
    print(f"\n[1/3] Loading dataset from: {DATA_DIR}")
    X, y, feature_names = load_dataset(DATA_DIR)
    print(f"  Dataset shape: {X.shape}  |  HR range: {y.min():.1f} – {y.max():.1f} bpm")

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "KNN (K=10)":        KNeighborsRegressor(n_neighbors=10),
        "Random Forest":     RandomForestRegressor(n_estimators=100,
                                                   random_state=RAND,
                                                   n_jobs=-1),
    }

    # Run CV
    print(f"\n[2/3] Running {N_FOLDS}-Fold Cross-Validation ...")
    results = run_cv(X, y, models, n_folds=N_FOLDS, seed=RAND)

    # Summarise
    rows = []
    print("\n" + "=" * 60)
    print(f"{'Model':<22} {'CV MAE (mean±std)':<22} {'CV RMSE (mean±std)'}")
    print("-" * 60)
    for name, vals in results.items():
        mae_arr  = np.array(vals["mae"])
        rmse_arr = np.array(vals["rmse"])
        row = {
            "Model":     name,
            "MAE_mean":  round(mae_arr.mean(),  3),
            "MAE_std":   round(mae_arr.std(),   3),
            "RMSE_mean": round(rmse_arr.mean(), 3),
            "RMSE_std":  round(rmse_arr.std(),  3),
        }
        rows.append(row)
        print(f"  {name:<20} {mae_arr.mean():.2f} ± {mae_arr.std():.2f} bpm"
              f"       {rmse_arr.mean():.2f} ± {rmse_arr.std():.2f} bpm")
    print("=" * 60)

    summary_df = pd.DataFrame(rows)

    # Save CSVs
    summary_df.to_csv(os.path.join(RESULT_DIR, "rq1_cv_summary.csv"), index=False)
    print(f"\n  Summary saved → {RESULT_DIR}/rq1_cv_summary.csv")

    # Per-fold CSV
    fold_rows = []
    for name, vals in results.items():
        for fold_i, (m, r) in enumerate(zip(vals["mae"], vals["rmse"]), 1):
            fold_rows.append({"Model": name, "Fold": fold_i, "MAE": m, "RMSE": r})
    pd.DataFrame(fold_rows).to_csv(
        os.path.join(RESULT_DIR, "rq1_cv_per_fold.csv"), index=False
    )

    # Plots
    print("\n[3/3] Generating figures ...")
    plot_metric(summary_df, "MAE_mean", "MAE_std",
                "MAE [bpm]",
                "5-Fold CV — Mean Absolute Error (MAE) — Resting-State Models",
                os.path.join(FIG_DIR, "rq1_cv_mae_comparison.png"))

    plot_metric(summary_df, "RMSE_mean", "RMSE_std",
                "RMSE [bpm]",
                "5-Fold CV — Root Mean Square Error (RMSE) — Resting-State Models",
                os.path.join(FIG_DIR, "rq1_cv_rmse_comparison.png"))

    plot_combined(summary_df,
                  os.path.join(FIG_DIR, "rq1_cv_combined.png"))

    print("\n✅  All done!")
    print(f"   Results → {RESULT_DIR}/rq1_cv_summary.csv")
    print(f"   Figures → {FIG_DIR}/rq1_cv_*.png")


if __name__ == "__main__":
    main()
