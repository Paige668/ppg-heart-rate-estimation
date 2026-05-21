"""Task 6: Extract 20 features from all 1337 PPG windows → results/features.csv"""
import os, glob, warnings
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ppg_10s_windows_rest")
RES_DIR  = "results"
FS       = 25.0   # Hz
N        = 250    # samples per 10-second window (10 * 25)

os.makedirs(RES_DIR, exist_ok=True)
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"Total windows: {len(all_files)}")


def detect_peaks(sig):
    min_dist   = int(FS * 60 / 100)   # max 100 bpm → 15 samples
    prominence = sig.std() * 0.3
    peaks, _   = signal.find_peaks(sig, distance=min_dist, prominence=prominence)
    return peaks


def extract_features(sig, fs=FS):
    feats = {}

    # ── Category 1: Statistical (7) ─────────────────────────────────────────
    feats["mean"]       = np.mean(sig)
    feats["std"]        = np.std(sig)
    feats["skewness"]   = float(stats.skew(sig))
    feats["kurtosis"]   = float(stats.kurtosis(sig))
    feats["min"]        = np.min(sig)
    feats["max"]        = np.max(sig)
    feats["mad"]        = np.mean(np.abs(sig - np.mean(sig)))

    # ── Category 2: Morphological / Time-domain (5) ─────────────────────────
    feats["peak_to_peak"] = feats["max"] - feats["min"]

    peaks = detect_peaks(sig)
    feats["n_peaks"] = len(peaks)

    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs          # RR intervals in seconds
        feats["rr_mean"]  = np.mean(rr)
        feats["rr_std"]   = np.std(rr)
        rr_diff = np.diff(rr)
        feats["rmssd"]    = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0.0
    else:
        feats["rr_mean"] = np.nan
        feats["rr_std"]  = np.nan
        feats["rmssd"]   = np.nan

    # ── Category 3: Frequency-domain (4) ────────────────────────────────────
    n      = len(sig)
    freqs  = fftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(fft(sig))[:n // 2]
    freqs   = freqs[:n // 2]

    # band mask: 0.5–4 Hz (30–240 bpm)
    band_mask = (freqs >= 0.5) & (freqs <= 4.0)
    band_power = np.sum(fft_mag[band_mask] ** 2)
    feats["band_power"] = band_power

    if band_mask.any():
        dominant_idx          = np.argmax(fft_mag[band_mask])
        dominant_freq         = freqs[band_mask][dominant_idx]
        dominant_amplitude    = fft_mag[band_mask][dominant_idx]
    else:
        dominant_freq      = np.nan
        dominant_amplitude = np.nan

    feats["dominant_freq"]      = dominant_freq
    feats["dominant_amplitude"] = dominant_amplitude

    # Spectral entropy over the full PSD
    psd  = fft_mag ** 2
    psd_norm = psd / (psd.sum() + 1e-12)
    feats["spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    # ── Category 4: Additional statistical / signal quality (2) ─────────────
    feats["energy"] = np.sum(sig ** 2) / n

    # Zero-crossing rate
    zero_crossings    = np.diff(np.sign(sig - np.mean(sig)))
    feats["zcr"]      = np.sum(zero_crossings != 0) / n

    # ── Category 5: New morphological / Hjorth features (2) ──────────────────
    # Hjorth Mobility: proxy for mean frequency via time-domain derivatives.
    # Provides frequency-rate information independently of FFT.
    dsig = np.diff(sig)
    feats["hjorth_mobility"] = np.sqrt(np.var(dsig) / (np.var(sig) + 1e-12))

    # Fall time: mean duration from systolic peak to next trough (seconds).
    # Captures pulse-shape information independently of beat rate.
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


# ── Run extraction ────────────────────────────────────────────────────────────
records = []
for i, f in enumerate(all_files):
    df   = pd.read_csv(f)
    sig  = df["PPG_filtered"].values
    hr   = df["HR"].iloc[0]
    feat = extract_features(sig)
    feat["HR"]   = hr
    feat["file"] = os.path.basename(f)
    records.append(feat)
    if (i + 1) % 200 == 0:
        print(f"  Processed {i+1}/{len(all_files)} ...")

features_df = pd.DataFrame(records)

# Reorder: file first, HR last
cols = ["file"] + [c for c in features_df.columns if c not in ("file", "HR")] + ["HR"]
features_df = features_df[cols]

out = os.path.join(RES_DIR, "features.csv")
features_df.to_csv(out, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
feature_cols = [c for c in features_df.columns if c not in ("file", "HR")]
print(f"\nFeature extraction complete.")
print(f"  Shape        : {features_df.shape[0]} rows × {len(feature_cols)} features (20 features)")
print(f"  NaN counts   :\n{features_df[feature_cols].isna().sum()[features_df[feature_cols].isna().sum() > 0]}")
print(f"\nFeature statistics:")
print(features_df[feature_cols].describe().round(4).to_string())
print(f"\nSaved: {out}")
