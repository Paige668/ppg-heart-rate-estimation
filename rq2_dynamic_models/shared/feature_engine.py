import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq

def _detect_peaks(sig, fs):
    min_dist   = int(fs * 60 / 100)
    prominence = sig.std() * 0.3
    peaks, _   = signal.find_peaks(sig, distance=min_dist, prominence=prominence)
    return peaks

def extract_matched_features(ppg_signal, acc_x, acc_y, acc_z, fs=125.0):
    """
    Extracts a strict matched feature set from both PPG and Accelerometer data.
    This version uses the exact same 20 PPG features as the original baselines,
    plus 12 Acc features (4 per axis).
    """
    feats = {}

    # --- 1. PPG Features (Exact 20 from original baseline) ---
    feats["ppg_mean"]     = np.mean(ppg_signal)
    feats["ppg_std"]      = np.std(ppg_signal)
    feats["ppg_skewness"] = float(stats.skew(ppg_signal))
    feats["ppg_kurtosis"] = float(stats.kurtosis(ppg_signal))
    feats["ppg_min"]      = np.min(ppg_signal)
    feats["ppg_max"]      = np.max(ppg_signal)
    feats["ppg_mad"]      = np.mean(np.abs(ppg_signal - np.mean(ppg_signal)))

    feats["ppg_peak_to_peak"] = feats["ppg_max"] - feats["ppg_min"]
    peaks = _detect_peaks(ppg_signal, fs)
    feats["ppg_n_peaks"] = len(peaks)
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        feats["ppg_rr_mean"] = np.mean(rr)
        feats["ppg_rr_std"]  = np.std(rr)
        rr_diff = np.diff(rr)
        feats["ppg_rmssd"]   = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0.0
    else:
        feats["ppg_rr_mean"] = 0.0
        feats["ppg_rr_std"]  = 0.0
        feats["ppg_rmssd"]   = 0.0

    n       = len(ppg_signal)
    freqs   = fftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(fft(ppg_signal))[:n // 2]
    freqs   = freqs[:n // 2]
    band_mask  = (freqs >= 0.5) & (freqs <= 4.0)
    band_power = np.sum(fft_mag[band_mask] ** 2)
    feats["ppg_band_power"] = band_power
    if band_mask.any():
        dominant_idx = np.argmax(fft_mag[band_mask])
        feats["ppg_dominant_freq"]      = freqs[band_mask][dominant_idx]
        feats["ppg_dominant_amplitude"] = fft_mag[band_mask][dominant_idx]
    else:
        feats["ppg_dominant_freq"]      = 0.0
        feats["ppg_dominant_amplitude"] = 0.0
    psd      = fft_mag ** 2
    psd_norm = psd / (psd.sum() + 1e-12)
    feats["ppg_spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

    feats["ppg_energy"] = np.sum(ppg_signal ** 2) / n
    zero_crossings  = np.diff(np.sign(ppg_signal - np.mean(ppg_signal)))
    feats["ppg_zcr"]    = np.sum(zero_crossings != 0) / n

    dsig = np.diff(ppg_signal)
    feats["ppg_hjorth_mobility"] = np.sqrt(np.var(dsig) / (np.var(ppg_signal) + 1e-12))
    troughs, _ = signal.find_peaks(-ppg_signal, distance=int(fs * 60 / 100),
                                   prominence=ppg_signal.std() * 0.3)
    if len(peaks) >= 1 and len(troughs) >= 1:
        fall_times = []
        for p in peaks:
            after = troughs[troughs > p]
            if len(after):
                fall_times.append((after[0] - p) / fs)
        feats["ppg_fall_time"] = np.mean(fall_times) if fall_times else 0.0
    else:
        feats["ppg_fall_time"] = 0.0

    # --- 2. Acc Features (12: 4 per axis) ---
    for name, sig_data in [('acc_x', acc_x), ('acc_y', acc_y), ('acc_z', acc_z)]:
        feats[f"{name}_mean"]   = np.mean(sig_data)
        feats[f"{name}_std"]    = np.std(sig_data)
        feats[f"{name}_energy"] = np.sum(sig_data ** 2) / len(sig_data)
        
        a_n = len(sig_data)
        a_freqs = fftfreq(a_n, d=1.0 / fs)
        a_fft_mag = np.abs(fft(sig_data))[:a_n // 2]
        a_freqs = a_freqs[:a_n // 2]
        a_band_mask = (a_freqs >= 0.5) & (a_freqs <= 10.0)
        if a_band_mask.any():
            feats[f"{name}_dom_freq"] = a_freqs[a_band_mask][np.argmax(a_fft_mag[a_band_mask])]
        else:
            feats[f"{name}_dom_freq"] = 0.0

    return feats

def get_feature_names():
    return [
        "ppg_mean", "ppg_std", "ppg_skewness", "ppg_kurtosis", "ppg_min", "ppg_max", "ppg_mad",
        "ppg_peak_to_peak", "ppg_n_peaks", "ppg_rr_mean", "ppg_rr_std", "ppg_rmssd",
        "ppg_band_power", "ppg_dominant_freq", "ppg_dominant_amplitude", "ppg_spectral_entropy",
        "ppg_energy", "ppg_zcr", "ppg_hjorth_mobility", "ppg_fall_time",
        "acc_x_mean", "acc_x_std", "acc_x_energy", "acc_x_dom_freq",
        "acc_y_mean", "acc_y_std", "acc_y_energy", "acc_y_dom_freq",
        "acc_z_mean", "acc_z_std", "acc_z_energy", "acc_z_dom_freq"
    ]
