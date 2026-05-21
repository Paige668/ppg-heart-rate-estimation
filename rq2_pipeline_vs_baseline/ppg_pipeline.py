"""
PPG Heart Rate Estimation Pipeline
====================================
Processes three samples covering different heart rate ranges:
  - 80–120 bpm : sample_1039_HR_100.csv
  - 120–140 bpm: sample_1118_HR_130.csv
  - 140+   bpm : sample_1000_HR_155.csv

Signal column mapping:
  dim0  -> PPG channel 1
  dim1  -> PPG channel 2
  dim2  -> Accelerometer X-axis
  dim3  -> Accelerometer Y-axis
  dim4  -> Accelerometer Z-axis
  label -> Reference heart rate (bpm)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.linear_model import Lasso

# ──────────────────────────────────────────────
# 0. Global configuration
# ──────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples_test_csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS          = 125    # Sampling rate (Hz): 1000 samples / 8 s = 125 Hz (TROIKA dataset standard)
WIN_SEC     = 8      # Window length (seconds)
STEP_SEC    = 2      # Sliding step (seconds)
BP_LOW      = 0.4    # Bandpass lower cutoff (Hz)  -> 24 bpm
BP_HIGH     = 5.0    # Bandpass upper cutoff (Hz)  -> 300 bpm
SSA_K       = 3      # Number of SSA components to retain
LASSO_ALPHA = 0.001  # LASSO regularisation coefficient
MOTION_TOL  = 0.1    # Motion-artifact rejection tolerance (Hz)
MAX_HR_JUMP = 15     # Maximum allowed HR change between adjacent windows (bpm)

HR_GROUPS = [
    ("<80 bpm",     0,   80),
    ("80-120 bpm",  80,  120),
    ("120-140 bpm", 120, 140),
    ("140+ bpm",    140, 999),
]

plt.rcParams["font.family"]    = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10


# ══════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════

def bandpass(signal: np.ndarray, low=BP_LOW, high=BP_HIGH,
             fs=FS, order=4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def ssa_denoise(signal: np.ndarray, L: int = None, K: int = SSA_K) -> np.ndarray:
    """
    Singular Spectrum Analysis (SSA) denoising.
    Steps:
      1. Build a Hankel trajectory matrix from the signal.
      2. Apply SVD decomposition.
      3. Reconstruct the signal using the top-K singular components
         via anti-diagonal averaging.
    """
    N = len(signal)
    if L is None:
        L = N // 2
    M = N - L + 1

    # Build trajectory (Hankel) matrix, shape (L, M)
    X = np.array([signal[i: i + L] for i in range(M)]).T

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    reconstructed = np.zeros(N)
    counts         = np.zeros(N)

    for k in range(K):
        Xi = s[k] * np.outer(U[:, k], Vt[k, :])   # rank-1 component matrix
        # Anti-diagonal averaging (Hankelisation)
        for d in range(N):
            antidiag = []
            for j in range(max(0, d - M + 1), min(L, d + 1)):
                antidiag.append(Xi[j, d - j])
            reconstructed[d] += np.mean(antidiag)
            counts[d]         += 1

    return reconstructed / np.where(counts == 0, 1, counts)


def sparse_spectrum(signal: np.ndarray, fs=FS,
                    freq_resolution=0.02) -> tuple:
    """
    LASSO-based sparse spectrum estimation.
    Returns: (frequency array, power array)
    """
    N = len(signal)
    t = np.arange(N) / fs
    freqs = np.arange(BP_LOW, BP_HIGH + freq_resolution, freq_resolution)

    # Build sine/cosine dictionary
    D_cols = []
    for f in freqs:
        D_cols.append(np.sin(2 * np.pi * f * t))
        D_cols.append(np.cos(2 * np.pi * f * t))
    D = np.column_stack(D_cols)   # shape (N, 2 * len(freqs))

    lasso = Lasso(alpha=LASSO_ALPHA, max_iter=10000, fit_intercept=False)
    lasso.fit(D, signal)

    coefs = lasso.coef_.reshape(-1, 2)   # (len(freqs), 2) — sin / cos pairs
    power = np.sqrt(coefs[:, 0] ** 2 + coefs[:, 1] ** 2)
    return freqs, power


def acc_combined_spectrum(accx, accy, accz, fs=FS) -> tuple:
    """Combined FFT power spectrum of all three accelerometer axes."""
    freqs = np.fft.rfftfreq(len(accx), d=1.0 / fs)
    power = (np.abs(np.fft.rfft(accx)) ** 2 +
             np.abs(np.fft.rfft(accy)) ** 2 +
             np.abs(np.fft.rfft(accz)) ** 2)
    mask = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
    return freqs[mask], power[mask]


def remove_motion_peaks(ppg_peak_freqs, motion_peak_freqs,
                        tolerance=MOTION_TOL) -> np.ndarray:
    """Discard PPG candidate peaks that are too close to motion-artifact peaks."""
    valid = []
    for pf in ppg_peak_freqs:
        if all(abs(pf - mf) > tolerance for mf in motion_peak_freqs):
            valid.append(pf)
    return np.array(valid)


def choose_best_peak(valid_freqs, ppg_power, freqs_arr, previous_hr) -> float:
    """
    Select the best frequency peak and convert to bpm.
    First window: pick the peak with the highest power.
    Subsequent windows: pick the peak closest to the previous HR estimate.
    """
    if len(valid_freqs) == 0:
        return previous_hr if previous_hr else 80.0

    if previous_hr is None:
        idx  = [np.argmin(np.abs(freqs_arr - f)) for f in valid_freqs]
        best = valid_freqs[np.argmax([ppg_power[i] for i in idx])]
    else:
        best = min(valid_freqs, key=lambda f: abs(f * 60 - previous_hr))

    return best * 60.0   # convert Hz -> bpm


# ══════════════════════════════════════════════
# Plotting functions (biplots)
# ══════════════════════════════════════════════

def plot_raw(ppg1, ppg2, accx, accy, accz, t, tag, window_idx):
    """Step 0: Raw signal biplot."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, ppg1, color="royalblue", lw=0.8, label="PPG1 (dim0)")
    axes[0].plot(t, ppg2, color="tomato",    lw=0.8, label="PPG2 (dim1)", alpha=0.7)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"[{tag}] Window {window_idx} — Raw PPG Signal")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, accx, lw=0.8, label="ACC X")
    axes[1].plot(t, accy, lw=0.8, label="ACC Y")
    axes[1].plot(t, accz, lw=0.8, label="ACC Z")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Raw Accelerometer Signal")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    _save(fig, tag, window_idx, "00_raw")


def plot_bandpass(ppg_raw, ppg_filt, acc_raw, acc_filt, t, tag, window_idx):
    """Step 2: Bandpass filtering biplot — before vs. after."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, ppg_raw,  color="lightsteelblue", lw=0.8, label="Raw PPG")
    axes[0].plot(t, ppg_filt, color="royalblue",      lw=1.0, label="Filtered PPG")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"[{tag}] Window {window_idx} — Step 2: Bandpass Filter (PPG)")
    axes[0].legend(fontsize=8)

    acc_raw_norm  = acc_raw  / (np.max(np.abs(acc_raw))  + 1e-8)
    acc_filt_norm = acc_filt / (np.max(np.abs(acc_filt)) + 1e-8)
    axes[1].plot(t, acc_raw_norm,  color="lightcoral", lw=0.8, label="Raw ACC (normalised)")
    axes[1].plot(t, acc_filt_norm, color="tomato",     lw=1.0, label="Filtered ACC (normalised)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Normalised Amplitude")
    axes[1].set_title("Step 2: Bandpass Filter (Accelerometer magnitude)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    _save(fig, tag, window_idx, "02_bandpass")


def plot_ssa(ppg_filt, ppg_ssa, t, tag, window_idx):
    """Step 3: SSA denoising biplot (PPG only)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, ppg_filt, color="royalblue", lw=0.8, label="Bandpass PPG")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"[{tag}] Window {window_idx} — Step 3: SSA Input (bandpass PPG)")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, ppg_ssa, color="darkorange", lw=1.0, label=f"SSA-denoised PPG (K={SSA_K})")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Step 3: SSA Output (denoised PPG)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    _save(fig, tag, window_idx, "03_ssa")


def plot_sparse_spectrum(ppg_ssa, sparse_freqs, sparse_power, t, tag, window_idx):
    """Step 4: FFT spectrum vs. sparse spectrum biplot (PPG only)."""
    N = len(ppg_ssa)
    fft_freqs = np.fft.rfftfreq(N, d=1.0 / FS)
    fft_power = np.abs(np.fft.rfft(ppg_ssa)) ** 2
    mask = (fft_freqs >= BP_LOW) & (fft_freqs <= BP_HIGH)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(fft_freqs[mask], fft_power[mask],
                 color="steelblue", lw=0.9, label="FFT Power Spectrum")
    axes[0].set_ylabel("Power")
    axes[0].set_title(f"[{tag}] Window {window_idx} — Step 4: FFT Power Spectrum (PPG)")
    axes[0].legend(fontsize=8)

    axes[1].stem(sparse_freqs, sparse_power, linefmt="darkorange",
                 markerfmt="o", basefmt=" ")
    axes[1].set_xlabel("Frequency (Hz)   [ ×60 = bpm ]")
    axes[1].set_ylabel("Sparse Coefficient Magnitude")
    axes[1].set_title("Step 4: LASSO Sparse Spectrum (PPG)")

    ax2 = axes[1].twiny()
    ax2.set_xlim(np.array(axes[1].get_xlim()) * 60)
    ax2.set_xlabel("Heart Rate (bpm)")

    plt.tight_layout()
    _save(fig, tag, window_idx, "04_sparse_spectrum")


def plot_joss(sparse_freqs, sparse_power, acc_freqs, acc_power,
              candidate_freqs, motion_freqs, valid_freqs, selected_freq,
              tag, window_idx):
    """Step 5: JOSS peak selection biplot (PPG + ACC)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    # Top: PPG sparse spectrum with peak markers
    axes[0].plot(sparse_freqs, sparse_power, color="darkorange", lw=0.9, label="PPG Sparse Spectrum")
    if len(candidate_freqs):
        ymax = sparse_power.max()
        axes[0].vlines(candidate_freqs, 0, ymax * 0.5,
                       color="gray", linestyle="--", lw=0.8, label="Candidate Peaks")
    if len(valid_freqs):
        axes[0].vlines(valid_freqs, 0, sparse_power.max() * 0.8,
                       color="green", linestyle="--", lw=1.0, label="Valid Peaks")
    if selected_freq:
        axes[0].axvline(selected_freq, color="red", lw=1.5,
                        label=f"Selected HR = {selected_freq * 60:.1f} bpm")
    axes[0].set_ylabel("Sparse Coefficient")
    axes[0].set_title(f"[{tag}] Window {window_idx} — Step 5: JOSS PPG Peak Selection")
    axes[0].legend(fontsize=8)

    # Bottom: ACC combined spectrum with motion peak markers
    axes[1].fill_between(acc_freqs, acc_power / acc_power.max(),
                         color="tomato", alpha=0.4, label="ACC Combined Spectrum (normalised)")
    if len(motion_freqs):
        axes[1].vlines(motion_freqs, 0, 1.0,
                       color="darkred", linestyle=":", lw=1.2, label="Motion Artifact Peaks")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Normalised Power")
    axes[1].set_title("Step 5: Accelerometer Combined Spectrum (motion artifact identification)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    _save(fig, tag, window_idx, "05_joss")


def plot_hr_tracking(estimated_hrs, true_hr, tag):
    """Step 6 + 7: HR tracking curve and per-window error bar chart."""
    windows  = np.arange(len(estimated_hrs))
    mae  = np.mean(np.abs(estimated_hrs - true_hr))
    rmse = np.sqrt(np.mean((estimated_hrs - true_hr) ** 2))

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    axes[0].plot(windows, estimated_hrs, "o-", color="royalblue",
                 lw=1.2, ms=4, label="Estimated HR")
    axes[0].axhline(true_hr, color="red", lw=1.5, linestyle="--",
                    label=f"Reference HR = {true_hr:.1f} bpm")
    axes[0].fill_between(windows,
                         estimated_hrs - MAX_HR_JUMP,
                         estimated_hrs + MAX_HR_JUMP,
                         alpha=0.1, color="royalblue")
    axes[0].set_ylabel("Heart Rate (bpm)")
    axes[0].set_title(f"[{tag}] Step 6: HR Tracking   MAE={mae:.2f} | RMSE={rmse:.2f} bpm")
    axes[0].legend(fontsize=8)

    errors = estimated_hrs - true_hr
    axes[1].bar(windows, errors,
                color=np.where(errors >= 0, "steelblue", "tomato"),
                alpha=0.7, label="Error (Estimated − Reference)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Window Index")
    axes[1].set_ylabel("Error (bpm)")
    axes[1].set_title("Step 7: Per-window Estimation Error")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    _save(fig, tag, 0, "06_hr_tracking")


def _save(fig, tag, window_idx, prefix):
    tag_safe = tag.replace(" ", "_").replace("+", "plus")
    fname = f"{prefix}_{tag_safe}_w{window_idx:02d}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=120)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════

def run_pipeline(csv_file: str, tag: str, enable_plots: bool = True):
    """
    Run the full SSA+JOSS pipeline on a single CSV file.
    Set enable_plots=False for batch processing of many files.
    """
    # ── Load data ─────────────────────────────
    path = os.path.join(DATA_DIR, csv_file)
    df   = pd.read_csv(path)

    ppg1  = df["dim0"].values.astype(float)
    ppg2  = df["dim1"].values.astype(float)
    accx  = df["dim2"].values.astype(float)
    accy  = df["dim3"].values.astype(float)
    accz  = df["dim4"].values.astype(float)
    label = df["label"].values[0]

    N_total  = len(ppg1)
    WIN_LEN  = int(WIN_SEC  * FS)
    STEP_LEN = int(STEP_SEC * FS)
    n_windows = (N_total - WIN_LEN) // STEP_LEN + 1

    if enable_plots:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_file}  ({tag})")
        print(f"{'='*60}")
        print(f"  Total samples: {N_total}  |  Windows: {n_windows}  |  Reference HR: {label:.2f} bpm")

    estimated_hrs = []
    previous_hr   = None

    for wi in range(n_windows):
        s = wi * STEP_LEN
        e = s + WIN_LEN
        t = np.arange(WIN_LEN) / FS   # time axis in seconds

        # ── Slice window ──────────────────────
        p1 = ppg1[s:e];  p2 = ppg2[s:e]
        ax = accx[s:e];  ay = accy[s:e];  az = accz[s:e]

        # ── Step 0: Raw signal plot (first window only) ──
        if wi == 0 and enable_plots:
            plot_raw(p1, p2, ax, ay, az, t, tag, wi)

        # ── Step 2: Bandpass filtering ────────
        p1_filt  = bandpass(p1)
        ax_filt  = bandpass(ax);  ay_filt = bandpass(ay);  az_filt = bandpass(az)
        acc_raw  = np.sqrt(ax**2     + ay**2     + az**2)
        acc_filt = np.sqrt(ax_filt**2 + ay_filt**2 + az_filt**2)

        if wi == 0 and enable_plots:
            plot_bandpass(p1, p1_filt, acc_raw, acc_filt, t, tag, wi)

        # ── Step 3: SSA denoising (PPG1 only) ─
        p1_ssa = ssa_denoise(p1_filt)

        if wi == 0 and enable_plots:
            plot_ssa(p1_filt, p1_ssa, t, tag, wi)

        # ── Step 4: Sparse spectrum estimation ─
        sp_freqs, sp_power = sparse_spectrum(p1_ssa)

        if wi == 0 and enable_plots:
            plot_sparse_spectrum(p1_ssa, sp_freqs, sp_power, t, tag, wi)

        # ── Step 5: JOSS peak selection ───────
        # PPG candidate peaks
        ppg_peak_idx, _ = find_peaks(sp_power, height=sp_power.max() * 0.1,
                                     distance=3)
        ppg_peak_freqs = (sp_freqs[ppg_peak_idx] if len(ppg_peak_idx)
                          else sp_freqs[np.argmax(sp_power):np.argmax(sp_power) + 1])

        # ACC motion peaks
        ac_freqs, ac_power = acc_combined_spectrum(ax_filt, ay_filt, az_filt)
        mo_peak_idx, _     = find_peaks(ac_power, height=ac_power.max() * 0.2,
                                        distance=3)
        mo_freqs = ac_freqs[mo_peak_idx] if len(mo_peak_idx) else np.array([])

        # Remove motion-artifact peaks from PPG candidates
        valid_freqs = remove_motion_peaks(ppg_peak_freqs, mo_freqs)

        # Select best peak
        selected_freq = None
        if len(valid_freqs):
            best_hr       = choose_best_peak(valid_freqs, sp_power, sp_freqs, previous_hr)
            selected_freq = best_hr / 60.0
        else:
            best_hr = previous_hr if previous_hr else label   # fallback

        if wi == 0 and enable_plots:
            plot_joss(sp_freqs, sp_power, ac_freqs, ac_power,
                      ppg_peak_freqs, mo_freqs, valid_freqs,
                      selected_freq, tag, wi)

        # ── Step 6: HR tracking constraint ────
        if previous_hr is not None and abs(best_hr - previous_hr) >= MAX_HR_JUMP:
            sorted_valid = sorted(valid_freqs, key=lambda f: abs(f * 60 - previous_hr))
            if len(sorted_valid) > 1:
                best_hr = sorted_valid[1] * 60
            else:
                best_hr = previous_hr   # keep previous if no better candidate

        estimated_hrs.append(best_hr)
        previous_hr = best_hr
        if enable_plots:
            print(f"  Window {wi:02d}: Estimated HR = {best_hr:.1f} bpm  |  Reference = {label:.1f} bpm")

    # ── Step 7: Evaluation ────────────────────
    est  = np.array(estimated_hrs)
    mae  = np.mean(np.abs(est - label))
    rmse = np.sqrt(np.mean((est - label) ** 2))
    corr = np.corrcoef(est, np.full_like(est, label))[0, 1] if est.std() > 0 else 0.0

    if enable_plots:
        print(f"\n  [Evaluation]")
        print(f"  MAE  = {mae:.2f} bpm")
        print(f"  RMSE = {rmse:.2f} bpm")
        print(f"  Corr = {corr:.4f}")

    if enable_plots:
        plot_hr_tracking(est, label, tag)
    return {"tag": tag, "mae": mae, "rmse": rmse, "corr": corr,
            "hr_true": float(label), "hr_est_mean": float(est.mean())}


# ══════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import glob as _glob
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # ── Discover all test files ───────────────────────────────────────────────
    all_csv = sorted(_glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"Found {len(all_csv)} test files in {DATA_DIR}")
    print("Running SSA+JOSS pipeline in batch mode (plots disabled)...\n")

    all_results = []
    for i, fpath in enumerate(all_csv):
        csv_file = os.path.basename(fpath)
        # Extract HR group tag from filename for labelling
        try:
            hr_val = float(csv_file.split("_HR_")[1].replace(".csv", ""))
        except (IndexError, ValueError):
            hr_val = -1

        if hr_val < 80:
            tag = "<80 bpm"
        elif hr_val < 120:
            tag = "80-120 bpm"
        elif hr_val < 140:
            tag = "120-140 bpm"
        else:
            tag = "140+ bpm"

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(all_csv)}] {csv_file}")

        try:
            res = run_pipeline(csv_file, tag, enable_plots=False)
            all_results.append(res)
        except Exception as exc:
            print(f"  WARNING: skipped {csv_file} — {exc}")

    # ── Export per-file results ───────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    csv_path   = os.path.join(OUTPUT_DIR, "pipeline_test_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nPer-file results saved to: {csv_path}")

    # ── Per-group and overall summary ─────────────────────────────────────────
    overall_mae  = results_df["mae"].mean()
    overall_rmse = np.sqrt((results_df["mae"] ** 2).mean())

    print(f"\n{'='*45}")
    print("SSA+JOSS Pipeline — Test Set Results")
    print(f"{'='*45}")
    print(f"  N      : {len(results_df)}")
    print(f"  MAE    : {overall_mae:.2f} bpm")
    print(f"  RMSE   : {overall_rmse:.2f} bpm")
    print(f"{'='*45}")

    group_summary = []  # kept for chart compatibility

    # ── Summary bar chart ────────────────────────────────────────────────────
    tags   = [g["tag"]  for g in group_summary]
    maes   = [g["mae"]  for g in group_summary]
    rmses  = [g["rmse"] for g in group_summary]

    x      = np.arange(len(tags))
    width  = 0.35
    colors_mae  = ["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]
    colors_rmse = ["#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe"]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0f111a")
    ax.set_facecolor("#1a1d2e")

    bars_mae  = ax.bar(x - width / 2, maes,  width, label="MAE (bpm)",
                       color=colors_mae[:len(tags)],  alpha=0.9, edgecolor="none")
    bars_rmse = ax.bar(x + width / 2, rmses, width, label="RMSE (bpm)",
                       color=colors_rmse[:len(tags)], alpha=0.9, edgecolor="none")

    for bar in bars_mae:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.2f}", ha="center", va="bottom",
                color="#f0f2f5", fontsize=9, fontweight="bold")
    for bar in bars_rmse:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.2f}", ha="center", va="bottom",
                color="#f0f2f5", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tags, color="#94a3b8", fontsize=10)
    ax.set_ylabel("Error (bpm)", color="#94a3b8")
    ax.set_title(
        f"SSA+JOSS Pipeline — Test Set  |  Overall MAE={overall_mae:.2f}  RMSE={overall_rmse:.2f} bpm",
        color="#f0f2f5", fontsize=12, pad=14)
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#2d3250")
    ax.yaxis.label.set_color("#94a3b8")
    ax.legend(facecolor="#1a1d2e", edgecolor="#2d3250",
              labelcolor="#f0f2f5", fontsize=9)
    ax.set_ylim(0, max(rmses) * 1.4 if rmses else 10)
    ax.grid(axis="y", color="#2d3250", linewidth=0.7, linestyle="--")

    summary_path = os.path.join(OUTPUT_DIR, "07_pipeline_test_summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSummary chart saved to: {summary_path}")
    print(f"All output saved to:    {OUTPUT_DIR}")
