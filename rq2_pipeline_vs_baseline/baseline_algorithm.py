import os
import glob
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Constants for test data ──────────────────────────────────────────────────
FS_TEST   = 125    # Sampling rate of test data (Hz)
BP_LOW    = 0.4    # Bandpass lower cutoff (Hz)
BP_HIGH   = 5.0    # Bandpass upper cutoff (Hz)

HR_GROUPS = [
    ("<80 bpm",    0,   80),
    ("80-120 bpm", 80,  120),
    ("120-140 bpm",120, 140),
    ("140+ bpm",   140, 999),
]


def bandpass_filter(sig, fs=FS_TEST, low=BP_LOW, high=BP_HIGH, order=4):
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def estimate_hr_from_peaks(sig, fs=FS_TEST, normalize=True):
    """
    PPG Heart Rate Estimation Baseline Algorithm (Peak-Based).

    Args:
        sig (array-like): The PPG signal (bandpass-filtered preferred).
        fs (float): Sampling frequency in Hz.
        normalize (bool): Whether to apply Z-score normalization.

    Returns:
        float: Estimated heart rate in bpm. Returns np.nan if < 2 peaks found.
    """
    if normalize:
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

    # min distance = 60% of the inter-peak interval at max HR (180 bpm)
    min_dist   = int(fs * 60 / 180)
    prominence = np.std(sig) * 0.3

    peaks, _ = signal.find_peaks(sig, distance=min_dist, prominence=prominence)

    if len(peaks) < 2:
        return np.nan

    rr_intervals = np.diff(peaks) / fs
    return 60.0 / np.mean(rr_intervals)


def run_baseline_on_test_directory(data_dir, output_file=None):
    """
    Run the peak-detection baseline on all CSV files in the test directory.

    Supports two CSV formats:
      - Legacy format : columns  PPG_filtered, HR   (fs=25 Hz)
      - Test format   : columns  dim0, label        (fs=125 Hz, raw PPG → bandpass first)

    Args:
        data_dir (str): Path to the directory containing PPG CSV files.
        output_file (str, optional): Path to save per-file results.

    Returns:
        pd.DataFrame: Results with columns file, HR_true, HR_estimated.
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    records   = []

    print(f"Processing {len(all_files)} files from {data_dir}...")

    for f in all_files:
        df    = pd.read_csv(f)
        fname = os.path.basename(f)

        # ── Detect format ────────────────────────────────────────────────────
        if "dim0" in df.columns and "label" in df.columns:
            # Test data: raw PPG → bandpass filter → peak detection at 125 Hz
            raw_ppg = df["dim0"].values.astype(float)
            sig     = bandpass_filter(raw_ppg, fs=FS_TEST)
            hr_gt   = float(df["label"].iloc[0])
            fs      = FS_TEST
        elif "PPG_filtered" in df.columns and "HR" in df.columns:
            # Legacy format: already filtered, 25 Hz
            sig   = df["PPG_filtered"].values.astype(float)
            hr_gt = float(df["HR"].iloc[0])
            fs    = 25.0
        else:
            continue

        hr_est = estimate_hr_from_peaks(sig, fs=fs, normalize=True)

        records.append({
            "file":         fname,
            "HR_true":      hr_gt,
            "HR_estimated": hr_est,
        })

    df_results = pd.DataFrame(records)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    valid = df_results.dropna(subset=["HR_estimated"])
    if len(valid) == 0:
        print("No valid results found.")
        return df_results

    # ── Overall metrics ──────────────────────────────────────────────────────
    mae_all  = mean_absolute_error(valid["HR_true"], valid["HR_estimated"])
    rmse_all = np.sqrt(mean_squared_error(valid["HR_true"], valid["HR_estimated"]))

    print(f"\n{'='*45}")
    print(f"Baseline (Peak Detection) — Test Set Results")
    print(f"{'='*45}")
    print(f"  N      : {len(valid)}")
    print(f"  MAE    : {mae_all:.2f} bpm")
    print(f"  RMSE   : {rmse_all:.2f} bpm")
    print(f"{'='*45}")

    return df_results


if __name__ == "__main__":
    BASE_DIR    = os.path.dirname(__file__)
    DATA_PATH   = os.path.join(os.path.dirname(BASE_DIR), "data", "samples_test_csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "output_plots", "baseline_test_results.csv")

    run_baseline_on_test_directory(DATA_PATH, OUTPUT_PATH)
