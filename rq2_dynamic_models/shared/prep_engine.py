import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def bandpass(sig, low=0.4, high=5.0, fs=125.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)

def ssa_denoise(sig, L=100, K=3):
    N = len(sig)
    M = N - L + 1
    X = np.array([sig[i: i + L] for i in range(M)]).T
    U, s_vals, Vt = np.linalg.svd(X, full_matrices=False)
    
    reconstructed = np.zeros(N)
    counts = np.zeros(N, dtype=int)
    row_idx, col_idx = np.indices((L, M))
    diag_idx = (row_idx + col_idx).ravel()
    np.add.at(counts, diag_idx, 1)
    
    for k in range(min(K, len(s_vals))):
        Xi = s_vals[k] * np.outer(U[:, k], Vt[k, :])
        np.add.at(reconstructed, diag_idx, Xi.ravel())
    
    return reconstructed / np.where(counts == 0, 1, counts)

def joss_filter(ppg_sig, ax, ay, az, fs=125.0):
    # Step 1: Bandpass
    ppg_bp = bandpass(ppg_sig, fs=fs)
    # Step 2: SSA
    ppg_ssa = ssa_denoise(ppg_bp, K=3)
    # Step 3: Minimal spectral subtraction (simplified mock or just return SSA)
    # In the full pipeline, JOSS actually picks peaks. 
    # For a strictly matched feature extraction, we want a cleaned SIGNAL.
    return ppg_ssa

def preprocess_signal(ppg, ax, ay, az, use_full_pipeline=False, fs=125.0):
    if not use_full_pipeline:
        return ppg
    else:
        # Full Pipeline: Bandpass + SSA
        # Note: We don't do JOSS peak-picking here because feature_engine needs the signal
        cleaned = bandpass(ppg, fs=fs)
        cleaned = ssa_denoise(cleaned, K=3)
        return cleaned
