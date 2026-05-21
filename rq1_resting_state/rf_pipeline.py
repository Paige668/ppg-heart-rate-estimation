"""
rf_pipeline.py — Standalone Random Forest pipeline for PPG heart rate estimation.

Usage
-----
# Train on one dataset folder, test on another:
python rf_pipeline.py --train /path/to/train_data --test /path/to/test_data

# Train on one folder only (80/20 split used for evaluation):
python rf_pipeline.py --train /path/to/train_data

# Save the trained model for later use:
python rf_pipeline.py --train /path/to/train_data --test /path/to/test_data --save-model model.pkl

# Load a previously saved model and evaluate on a new test set:
python rf_pipeline.py --test /path/to/test_data --load-model model.pkl
"""

import argparse
import glob
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FS = 25.0   # Expected sampling frequency (Hz)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _detect_peaks(sig, fs):
    min_dist   = int(fs * 60 / 100)    # max ~100 bpm
    prominence = sig.std() * 0.3
    peaks, _   = signal.find_peaks(sig, distance=min_dist, prominence=prominence)
    return peaks


def extract_features(sig, fs=FS):
    """Extract 20 features from a single PPG window (1-D numpy array)."""
    feats = {}

    # ── Category 1: Statistical (7) ──────────────────────────────────────────
    feats["mean"]     = np.mean(sig)
    feats["std"]      = np.std(sig)
    feats["skewness"] = float(stats.skew(sig))
    feats["kurtosis"] = float(stats.kurtosis(sig))
    feats["min"]      = np.min(sig)
    feats["max"]      = np.max(sig)
    feats["mad"]      = np.mean(np.abs(sig - np.mean(sig)))

    # ── Category 2: Morphological / Time-domain (5) ──────────────────────────
    feats["peak_to_peak"] = feats["max"] - feats["min"]

    peaks = _detect_peaks(sig, fs)
    feats["n_peaks"] = len(peaks)

    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        feats["rr_mean"] = np.mean(rr)
        feats["rr_std"]  = np.std(rr)
        rr_diff = np.diff(rr)
        feats["rmssd"]   = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0.0
    else:
        feats["rr_mean"] = np.nan
        feats["rr_std"]  = np.nan
        feats["rmssd"]   = np.nan

    # ── Category 3: Frequency-domain (4) ─────────────────────────────────────
    n       = len(sig)
    freqs   = fftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(fft(sig))[:n // 2]
    freqs   = freqs[:n // 2]

    band_mask  = (freqs >= 0.5) & (freqs <= 4.0)
    band_power = np.sum(fft_mag[band_mask] ** 2)
    feats["band_power"] = band_power

    if band_mask.any():
        dominant_idx       = np.argmax(fft_mag[band_mask])
        feats["dominant_freq"]      = freqs[band_mask][dominant_idx]
        feats["dominant_amplitude"] = fft_mag[band_mask][dominant_idx]
    else:
        feats["dominant_freq"]      = np.nan
        feats["dominant_amplitude"] = np.nan

    psd      = fft_mag ** 2
    psd_norm = psd / (psd.sum() + 1e-12)
    feats["spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    # ── Category 4: Additional statistical / signal quality (2) ──────────────
    feats["energy"] = np.sum(sig ** 2) / n

    zero_crossings = np.diff(np.sign(sig - np.mean(sig)))
    feats["zcr"]   = np.sum(zero_crossings != 0) / n

    # ── Category 5: Hjorth + Fall time (2) ───────────────────────────────────
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


def extract_features_from_folder(folder, fs=FS, ppg_col="PPG_filtered", hr_col="HR"):
    """
    Load all CSV files in `folder`, extract 20 features from each window.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 20)
    y : np.ndarray, shape (n_samples,)
    feature_names : list[str]
    file_names : list[str]
    """
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    print(f"  Found {len(csv_files)} CSV files in '{folder}'")

    records    = []
    hr_values  = []
    file_names = []

    for i, f in enumerate(csv_files):
        df  = pd.read_csv(f)
        sig = df[ppg_col].values.astype(float)
        hr  = float(df[hr_col].iloc[0])

        feat = extract_features(sig, fs=fs)
        records.append(feat)
        hr_values.append(hr)
        file_names.append(os.path.basename(f))

        if (i + 1) % 200 == 0:
            print(f"    Processed {i + 1}/{len(csv_files)} ...")

    feat_df       = pd.DataFrame(records)
    feature_names = list(feat_df.columns)

    # Fill NaN with column median (robust fallback)
    feat_df = feat_df.fillna(feat_df.median())

    X = feat_df.values
    y = np.array(hr_values)
    return X, y, feature_names, file_names


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(label, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  [{label}]  MAE = {mae:.4f} bpm   RMSE = {rmse:.4f} bpm")
    return mae, rmse


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_pipeline(train_folder, fs=FS, ppg_col="PPG_filtered", hr_col="HR"):
    """Extract features, scale, train RF. Returns (model_bundle, train_metrics)."""
    print("\n[1/3] Extracting features from training set ...")
    X_train_raw, y_train, feature_names, _ = extract_features_from_folder(
        train_folder, fs=fs, ppg_col=ppg_col, hr_col=hr_col
    )
    print(f"  Training set: {X_train_raw.shape[0]} samples × {X_train_raw.shape[1]} features")

    print("\n[2/3] Scaling features (StandardScaler fit on training set only) ...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    print("\n[3/3] Training Random Forest Regressor ...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    print("\n  Training set evaluation (in-sample, for reference):")
    mae_tr, rmse_tr = evaluate("Train", y_train, y_pred_train)

    model_bundle = {"rf": rf, "scaler": scaler, "feature_names": feature_names}
    return model_bundle, {"mae": mae_tr, "rmse": rmse_tr}


def test_pipeline(model_bundle, test_folder, fs=FS, ppg_col="PPG_filtered", hr_col="HR"):
    """Extract features from test set, scale with saved scaler, evaluate."""
    rf            = model_bundle["rf"]
    scaler        = model_bundle["scaler"]
    feature_names = model_bundle["feature_names"]

    print("\n[1/2] Extracting features from test set ...")
    X_test_raw, y_test, feat_names_test, file_names = extract_features_from_folder(
        test_folder, fs=fs, ppg_col=ppg_col, hr_col=hr_col
    )

    # Align columns in case of order mismatch
    feat_df_test = pd.DataFrame(X_test_raw, columns=feat_names_test)
    feat_df_test = feat_df_test.reindex(columns=feature_names, fill_value=0.0)
    X_test = scaler.transform(feat_df_test.values)

    print("\n[2/2] Predicting and evaluating ...")
    y_pred = rf.predict(X_test)
    mae, rmse = evaluate("Test", y_test, y_pred)

    results_df = pd.DataFrame({
        "file":    file_names,
        "HR_true": y_test,
        "HR_pred": y_pred,
        "error":   np.abs(y_test - y_pred),
    })
    return results_df, {"mae": mae, "rmse": rmse}


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Random Forest pipeline for PPG heart rate estimation (20 features)"
    )
    parser.add_argument("--train",      type=str, default=None,
                        help="Folder with training CSV files")
    parser.add_argument("--test",       type=str, default=None,
                        help="Folder with test CSV files")
    parser.add_argument("--fs",         type=float, default=25.0,
                        help="Sampling frequency in Hz (default: 25)")
    parser.add_argument("--ppg-col",    type=str, default="PPG_filtered",
                        help="Column name for the PPG signal (default: PPG_filtered)")
    parser.add_argument("--hr-col",     type=str, default="HR",
                        help="Column name for the HR label (default: HR)")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Path to save the trained model bundle (.pkl)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to a previously saved model bundle (.pkl)")
    parser.add_argument("--out-dir",    type=str, default="results",
                        help="Directory for output CSV results (default: results/)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model_bundle = None

    # ── Load existing model ───────────────────────────────────────────────────
    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        with open(args.load_model, "rb") as f:
            model_bundle = pickle.load(f)
        print("  Model loaded successfully.")

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.train:
        if model_bundle is not None:
            print("Warning: --load-model and --train both specified. "
                  "Ignoring --load-model and retraining.")

        if args.test is None:
            # No separate test set → internal 80/20 split
            print("\n[INFO] No --test folder provided. Using 80/20 internal split.")
            print("\n[1/3] Extracting features ...")
            X_raw, y, feature_names, file_names = extract_features_from_folder(
                args.train, fs=args.fs, ppg_col=args.ppg_col, hr_col=args.hr_col
            )
            X_tr_raw, X_te_raw, y_tr, y_te, idx_tr, idx_te = train_test_split(
                X_raw, y, np.arange(len(y)), test_size=0.2, random_state=42
            )
            print("\n[2/3] Scaling ...")
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr_raw)
            X_te   = scaler.transform(X_te_raw)

            print("\n[3/3] Training RF ...")
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)

            y_pred = rf.predict(X_te)
            print("\n  Internal test set evaluation:")
            evaluate("Test (80/20 split)", y_te, y_pred)

            model_bundle = {"rf": rf, "scaler": scaler, "feature_names": feature_names}

            results_df = pd.DataFrame({
                "file":    [file_names[i] for i in idx_te],
                "HR_true": y_te,
                "HR_pred": y_pred,
                "error":   np.abs(y_te - y_pred),
            })
            out_path = os.path.join(args.out_dir, "rf_predictions.csv")
            results_df.to_csv(out_path, index=False)
            print(f"\n  Predictions saved to: {out_path}")

        else:
            model_bundle, _ = train_pipeline(
                args.train, fs=args.fs, ppg_col=args.ppg_col, hr_col=args.hr_col
            )

    # ── Test on external test set ─────────────────────────────────────────────
    if args.test and args.train:
        results_df, metrics = test_pipeline(
            model_bundle, args.test,
            fs=args.fs, ppg_col=args.ppg_col, hr_col=args.hr_col
        )
        out_path = os.path.join(args.out_dir, "rf_predictions.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\n  Predictions saved to: {out_path}")

    # ── Save model ────────────────────────────────────────────────────────────
    if args.save_model and model_bundle is not None:
        with open(args.save_model, "wb") as f:
            pickle.dump(model_bundle, f)
        print(f"\n  Model saved to: {args.save_model}")

    if model_bundle is None and args.test is None:
        print("Nothing to do. Provide --train and/or --test. See --help for usage.")

    # ── Test only (with loaded model) ─────────────────────────────────────────
    if args.test and args.train is None and model_bundle is not None:
        results_df, metrics = test_pipeline(
            model_bundle, args.test,
            fs=args.fs, ppg_col=args.ppg_col, hr_col=args.hr_col
        )
        out_path = os.path.join(args.out_dir, "rf_predictions.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\n  Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()
