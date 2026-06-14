"""Component-level validation for SSA, LASSO, and JOSS stages.

This script generates small, reproducible validation checks that provide
evidence for three pipeline components:
  1. SSA preserves dominant cardiac structure under synthetic motion noise.
  2. LASSO converges under the chosen regularization and recovers sparse peaks.
  3. JOSS-style rejection preserves cardiac peaks when frequencies are separable
     and fails under overlap for the expected methodological reason.
"""

from __future__ import annotations

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.exceptions import ConvergenceWarning

from ppg_pipeline import (
    DATA_DIR,
    FS,
    WIN_SEC,
    SAMPLES,
    acc_combined_spectrum,
    bandpass,
    choose_best_peak,
    remove_motion_peaks,
    sparse_spectrum,
    ssa_denoise,
)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_plots", "component_validation")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "component_validation_results.json")


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def dominant_frequency(signal: np.ndarray, fs: int = FS) -> float:
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= 0.4) & (freqs <= 5.0)
    masked_freqs = freqs[mask]
    masked_power = power[mask]
    return float(masked_freqs[np.argmax(masked_power)])


def safe_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def root_mean_square_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def make_synthetic_components(
    heart_freq: float,
    motion_freq: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = WIN_SEC * FS
    time = np.arange(n_samples) / FS

    clean_heart = (
        1.0 * np.sin(2 * np.pi * heart_freq * time)
        + 0.25 * np.sin(2 * np.pi * 2 * heart_freq * time + 0.2)
    )
    motion = 0.75 * np.sin(2 * np.pi * motion_freq * time + 0.6)
    noise = rng.normal(0.0, noise_std, n_samples)
    mixed_ppg = clean_heart + motion + noise

    accx = motion + 0.05 * rng.normal(size=n_samples)
    accy = 0.85 * motion + 0.05 * rng.normal(size=n_samples)
    accz = 0.65 * motion + 0.05 * rng.normal(size=n_samples)
    return time, clean_heart, mixed_ppg, accx, accy, accz


def validate_ssa() -> dict:
    cases = [
        {"heart_freq": 1.60, "motion_freq": 2.20, "noise_std": 0.14, "seed": 11},
        {"heart_freq": 1.95, "motion_freq": 2.70, "noise_std": 0.16, "seed": 12},
        {"heart_freq": 2.30, "motion_freq": 3.10, "noise_std": 0.12, "seed": 13},
    ]
    case_results = []

    for index, case in enumerate(cases):
        time, clean_heart, mixed_ppg, _, _, _ = make_synthetic_components(**case)
        clean_ref = bandpass(clean_heart)
        raw_input = bandpass(mixed_ppg)
        ssa_output = ssa_denoise(raw_input)

        corr_before = safe_corrcoef(raw_input, clean_ref)
        corr_after = safe_corrcoef(ssa_output, clean_ref)
        rmse_before = root_mean_square_error(raw_input, clean_ref)
        rmse_after = root_mean_square_error(ssa_output, clean_ref)
        dominant_before = dominant_frequency(raw_input)
        dominant_after = dominant_frequency(ssa_output)

        case_results.append(
            {
                "case": index + 1,
                "heart_freq_hz": case["heart_freq"],
                "corr_before": corr_before,
                "corr_after": corr_after,
                "rmse_before": rmse_before,
                "rmse_after": rmse_after,
                "dominant_before_hz": dominant_before,
                "dominant_after_hz": dominant_after,
            }
        )

        if index == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time, raw_input, label="Bandpassed noisy PPG", alpha=0.75)
            ax.plot(time, ssa_output, label="SSA output", linewidth=1.2)
            ax.plot(time, clean_ref, label="Bandpassed clean heart reference", linewidth=1.0)
            ax.set_title("SSA validation example")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, "ssa_validation_example.png"), dpi=140)
            plt.close(fig)

    return {
        "cases": case_results,
        "mean_corr_before": float(np.mean([item["corr_before"] for item in case_results])),
        "mean_corr_after": float(np.mean([item["corr_after"] for item in case_results])),
        "mean_rmse_before": float(np.mean([item["rmse_before"] for item in case_results])),
        "mean_rmse_after": float(np.mean([item["rmse_after"] for item in case_results])),
    }


def load_first_window(sample_file: str) -> np.ndarray:
    frame = pd.read_csv(os.path.join(DATA_DIR, sample_file))
    win_len = WIN_SEC * FS
    ppg = frame["dim0"].values.astype(float)[:win_len]
    return ssa_denoise(bandpass(ppg))


def validate_lasso() -> dict:
    synthetic_cases = [
        {"heart_freq": 1.55, "motion_freq": 2.35, "noise_std": 0.08, "seed": 21},
        {"heart_freq": 2.05, "motion_freq": 2.85, "noise_std": 0.10, "seed": 22},
        {"heart_freq": 2.40, "motion_freq": 3.20, "noise_std": 0.08, "seed": 23},
    ]

    synthetic_results = []
    for index, case in enumerate(synthetic_cases):
        _, _, mixed_ppg, _, _, _ = make_synthetic_components(**case)
        input_signal = ssa_denoise(bandpass(mixed_ppg))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            freqs, power = sparse_spectrum(input_signal)
        top_frequency = float(freqs[int(np.argmax(power))])
        concentration = float(np.sum(np.sort(power)[-3:]) / np.sum(power)) if np.sum(power) > 0 else 0.0
        active_bins = int(np.sum(power > 0.05 * np.max(power))) if np.max(power) > 0 else 0

        synthetic_results.append(
            {
                "case": index + 1,
                "heart_freq_hz": case["heart_freq"],
                "top_frequency_hz": top_frequency,
                "absolute_error_hz": abs(top_frequency - case["heart_freq"]),
                "active_bins_above_5pct": active_bins,
                "top3_power_ratio": concentration,
                "convergence_warning_count": len(caught),
            }
        )

        if index == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.stem(freqs, power, linefmt="tab:orange", markerfmt=" ", basefmt=" ")
            ax.axvline(case["heart_freq"], color="green", linestyle="--", label="True heart freq")
            ax.set_title("LASSO sparse spectrum validation example")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Coefficient magnitude")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, "lasso_validation_spectrum.png"), dpi=140)
            plt.close(fig)

    real_window_results = []
    for tag, sample_file in SAMPLES.items():
        signal = load_first_window(sample_file)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            freqs, power = sparse_spectrum(signal)
        top_frequency = float(freqs[int(np.argmax(power))])
        real_window_results.append(
            {
                "tag": tag,
                "sample_file": sample_file,
                "top_frequency_hz": top_frequency,
                "convergence_warning_count": len(caught),
            }
        )

    return {
        "synthetic_cases": synthetic_results,
        "real_windows": real_window_results,
        "synthetic_warning_total": int(sum(item["convergence_warning_count"] for item in synthetic_results)),
        "real_warning_total": int(sum(item["convergence_warning_count"] for item in real_window_results)),
        "mean_top_frequency_error_hz": float(np.mean([item["absolute_error_hz"] for item in synthetic_results])),
        "mean_top3_power_ratio": float(np.mean([item["top3_power_ratio"] for item in synthetic_results])),
    }


def run_joss_case(heart_freq: float, motion_freq: float, noise_std: float, seed: int) -> dict:
    _, _, mixed_ppg, accx, accy, accz = make_synthetic_components(
        heart_freq=heart_freq,
        motion_freq=motion_freq,
        noise_std=noise_std,
        seed=seed,
    )
    ppg_for_spectrum = ssa_denoise(bandpass(mixed_ppg))
    sp_freqs, sp_power = sparse_spectrum(ppg_for_spectrum)
    ppg_peak_idx, _ = find_peaks(sp_power, height=sp_power.max() * 0.1, distance=3)
    ppg_peak_freqs = sp_freqs[ppg_peak_idx] if len(ppg_peak_idx) else np.array([])

    ax_filt = bandpass(accx)
    ay_filt = bandpass(accy)
    az_filt = bandpass(accz)
    acc_freqs, acc_power = acc_combined_spectrum(ax_filt, ay_filt, az_filt)
    motion_peak_idx, _ = find_peaks(acc_power, height=acc_power.max() * 0.2, distance=3)
    motion_peak_freqs = acc_freqs[motion_peak_idx] if len(motion_peak_idx) else np.array([])

    valid_freqs = remove_motion_peaks(ppg_peak_freqs, motion_peak_freqs)
    selected_hr = None
    if len(valid_freqs):
        selected_hr = choose_best_peak(valid_freqs, sp_power, sp_freqs, previous_hr=None)

    return {
        "heart_freq_hz": heart_freq,
        "motion_freq_hz": motion_freq,
        "ppg_peak_freqs_hz": [float(value) for value in ppg_peak_freqs],
        "motion_peak_freqs_hz": [float(value) for value in motion_peak_freqs],
        "valid_freqs_hz": [float(value) for value in valid_freqs],
        "selected_hr_bpm": float(selected_hr) if selected_hr is not None else None,
    }


def validate_joss() -> dict:
    separable_case = run_joss_case(heart_freq=1.50, motion_freq=2.00, noise_std=0.05, seed=31)
    overlap_case = run_joss_case(heart_freq=2.45, motion_freq=2.50, noise_std=0.05, seed=32)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, case, title in [
        (axes[0], separable_case, "Separable frequencies"),
        (axes[1], overlap_case, "Overlapping frequencies"),
    ]:
        ax.axvline(case["heart_freq_hz"], color="green", linestyle="--", label="True heart")
        ax.axvline(case["motion_freq_hz"], color="red", linestyle=":", label="True motion")
        for value in case["ppg_peak_freqs_hz"]:
            ax.axvline(value, color="tab:orange", alpha=0.5)
        for value in case["valid_freqs_hz"]:
            ax.axvline(value, color="tab:blue", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(1.2, 2.8)
    axes[0].set_ylabel("Peak markers")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUTPUT_DIR, "joss_validation_cases.png"), dpi=140)
    plt.close(fig)

    return {
        "separable_case": separable_case,
        "overlap_case": overlap_case,
        "separable_preserved_heart": any(abs(value - 1.50) <= 0.04 for value in separable_case["valid_freqs_hz"]),
        "separable_removed_motion": all(abs(value - 2.00) > 0.08 for value in separable_case["valid_freqs_hz"]),
        "overlap_removed_true_heart": not any(abs(value - 2.45) <= 0.04 for value in overlap_case["valid_freqs_hz"]),
    }


def main() -> None:
    ensure_output_dir()
    results = {
        "ssa": validate_ssa(),
        "lasso": validate_lasso(),
        "joss": validate_joss(),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("Component validation completed.")
    print(json.dumps(results, indent=2))
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()