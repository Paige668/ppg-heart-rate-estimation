"""
Inherited raw-robust CNN route v2 with a unified algorithm across both cases.

This route keeps the inherited-v2 calibrated spectral CNN, but enforces a
single shared configuration for `no-pre` and `full-pipeline`. The only
intentional difference between the two entry points is whether the raw signals
go through the preprocessing pipeline before the shared feature extractor.
"""

from dataclasses import dataclass
import glob
import os
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
TRAIN_FOLDER = os.path.join(REPO_ROOT, "data", "samples_train_csv")
TEST_FOLDER = os.path.join(REPO_ROOT, "data", "samples_test_csv")

PPG_COL = "dim0"
ACCX_COL = "dim2"
ACCY_COL = "dim3"
ACCZ_COL = "dim4"
HR_COL = "label"

FS = 125.0
BP_LOW = 0.4
BP_HIGH = 5.0
SSA_K = 3

BATCH_SIZE = 32
EPOCHS = 120
PATIENCE = 24
RANDOM_STATE = 42

DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class RouteConfig:
    model_name: str
    use_full_pipeline: bool
    feature_high: float
    lr: float
    motion_gate_scale: float
    attn_temperature: float
    delta_cap_bpm: float
    scale_cap: float
    bias_penalty: float
    over_penalty: float


def build_unified_config(model_name: str, use_full_pipeline: bool) -> RouteConfig:
    # One shared general algorithm definition for both cases; only preprocessing differs.
    return RouteConfig(
        model_name=model_name,
        use_full_pipeline=use_full_pipeline,
        feature_high=3.0,
        lr=6e-4,
        motion_gate_scale=0.45,
        attn_temperature=0.58,
        delta_cap_bpm=5.0,
        scale_cap=0.06,
        bias_penalty=0.40,
        over_penalty=0.18,
    )


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bandpass(sig, low=BP_LOW, high=BP_HIGH, fs=FS, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def ssa_denoise(sig, L=None, K=SSA_K):
    n = len(sig)
    if L is None:
        L = min(n // 2, 100)
    m = n - L + 1
    hankel = np.array([sig[i: i + L] for i in range(m)]).T
    u, s_vals, vt = np.linalg.svd(hankel, full_matrices=False)
    reconstructed = np.zeros(n)
    counts = np.zeros(n, dtype=int)
    row_idx, col_idx = np.indices((L, m))
    diag_idx = (row_idx + col_idx).ravel()
    np.add.at(counts, diag_idx, 1)
    for k in range(K):
        component = s_vals[k] * np.outer(u[:, k], vt[k, :])
        np.add.at(reconstructed, diag_idx, component.ravel())
    return reconstructed / np.where(counts == 0, 1, counts)


def preprocess_sample(ppg_raw, accx_raw, accy_raw, accz_raw):
    ppg_bp = bandpass(ppg_raw)
    accx_bp = bandpass(accx_raw)
    accy_bp = bandpass(accy_raw)
    accz_bp = bandpass(accz_raw)
    ppg_clean = ssa_denoise(ppg_bp)
    return np.stack([ppg_clean, accx_bp, accy_bp, accz_bp], axis=1).astype(np.float32)


def load_folder(folder, use_full_pipeline=False):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    X_list, y_list, names = [], [], []
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue
        if len(df) < 1000:
            continue

        ppg = df[PPG_COL].values[:1000].astype(np.float64)
        accx = df[ACCX_COL].values[:1000].astype(np.float64)
        accy = df[ACCY_COL].values[:1000].astype(np.float64)
        accz = df[ACCZ_COL].values[:1000].astype(np.float64)
        hr = float(df[HR_COL].iloc[0])

        if (not np.isfinite(hr) or np.any(~np.isfinite(ppg)) or np.any(~np.isfinite(accx)) or
                np.any(~np.isfinite(accy)) or np.any(~np.isfinite(accz))):
            continue

        if use_full_pipeline:
            try:
                sig = preprocess_sample(ppg, accx, accy, accz)
            except Exception:
                continue
        else:
            sig = np.stack([ppg, accx, accy, accz], axis=1).astype(np.float32)

        if np.any(~np.isfinite(sig)):
            continue

        X_list.append(sig.astype(np.float32))
        y_list.append(hr)
        names.append(os.path.basename(file_path))

    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.float32), names


def make_train_val_split_indices(y, split_seed=RANDOM_STATE, val_size=0.2):
    all_indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_size,
        random_state=split_seed,
    )
    return train_idx, val_idx


def _build_harmonic_feature(base_power, freqs):
    if len(freqs) < 2:
        return base_power.copy()
    step = freqs[1] - freqs[0]
    double_idx = np.round((2.0 * freqs - freqs[0]) / step).astype(int)
    harmonic = base_power.copy()
    valid = (double_idx >= 0) & (double_idx < len(freqs))
    harmonic[:, valid] = base_power[:, valid] + 0.5 * base_power[:, double_idx[valid]]
    return harmonic


def compute_inherited_features(X, f_high):
    time_len = X.shape[1]
    all_freqs = np.fft.rfftfreq(time_len, d=1.0 / FS)
    mask = (all_freqs >= BP_LOW) & (all_freqs <= f_high)
    freqs = all_freqs[mask]

    ppg = X[:, :, 0]
    acc = X[:, :, 1:4]

    ppg_power = np.abs(np.fft.rfft(ppg, axis=1))[:, mask] ** 2
    acc_power = np.abs(np.fft.rfft(acc, axis=1))[:, mask, :] ** 2
    acc_sum = acc_power.sum(axis=2)
    acc_max = acc_power.max(axis=2)

    log_ppg = np.log1p(ppg_power)
    log_acc = np.log1p(acc_sum)
    log_acc_max = np.log1p(acc_max)
    motion_suppressed = np.log1p(ppg_power / (1.0 + acc_sum + 0.25 * acc_max))
    motion_margin = log_ppg - log_acc
    harmonic_ppg = _build_harmonic_feature(log_ppg, freqs)
    harmonic_margin = harmonic_ppg - log_acc
    freq_chan = np.broadcast_to(freqs[np.newaxis, :], log_ppg.shape).astype(np.float32)

    features = np.stack(
        [
            log_ppg,
            log_acc,
            log_acc_max,
            motion_suppressed,
            motion_margin,
            harmonic_ppg,
            harmonic_margin,
            freq_chan,
        ],
        axis=2,
    ).astype(np.float32)
    return features, freqs.astype(np.float32)


def fit_scaler(X_train):
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def augment_features(x):
    sig = x.clone()
    spec = sig[:, :7]
    freq = sig[:, 7:]
    noise_std = spec.abs().mean() * 0.025
    spec = spec + torch.randn_like(spec) * noise_std
    scale = torch.empty(spec.shape[1]).uniform_(0.94, 1.06)
    spec = spec * scale
    return torch.cat([spec, freq], dim=1)


class SpectralDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = augment_features(x)
        return x, self.y[idx]


class InheritedRawRobustCNN(nn.Module):
    def __init__(self, freq_axis, config: RouteConfig):
        super().__init__()
        self.config = config
        self.register_buffer("freq_hz", torch.tensor(freq_axis, dtype=torch.float32))

        self.ppg_branch = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.motion_branch = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.motion_gate = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.Sigmoid(),
        )
        self.attn_head = nn.Conv1d(64, 1, kernel_size=1)
        self.scale_head = nn.Sequential(
            nn.Linear(64 + 32 + 1, 32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(64 + 32 + 1, 48),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        ppg_in = x[:, [0, 3, 4, 5, 6, 7], :]
        motion_in = x[:, [1, 2, 7], :]

        ppg_feat = self.ppg_branch(ppg_in)
        motion_feat = self.motion_branch(motion_in)
        gate = self.motion_gate(motion_feat)
        cleaned_feat = ppg_feat * (1.0 - self.config.motion_gate_scale * gate)

        logits = self.attn_head(cleaned_feat) / self.config.attn_temperature
        attn_w = torch.softmax(logits, dim=-1)
        freq = self.freq_hz.view(1, 1, -1)
        peak_hz = (attn_w * freq).sum(dim=-1).squeeze(-1)
        hr_attn = peak_hz * 60.0

        pooled = torch.cat(
            [
                torch.mean(cleaned_feat, dim=-1),
                torch.mean(motion_feat, dim=-1),
                (hr_attn / 200.0).unsqueeze(1),
            ],
            dim=1,
        )
        scale = 1.0 + self.config.scale_cap * torch.tanh(self.scale_head(pooled).squeeze(-1))
        delta = self.config.delta_cap_bpm * torch.tanh(self.delta_head(pooled).squeeze(-1))
        return hr_attn * scale + delta


def general_regression_loss(pred, target, config: RouteConfig):
    error = pred - target
    mae = torch.abs(error).mean()
    bias = torch.abs(error.mean())
    over = torch.relu(error).mean()
    return mae + config.bias_penalty * bias + config.over_penalty * over


def make_loader(X, y, shuffle=False, augment=False):
    ds = SpectralDataset(X, y, augment=augment)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_model(model, train_loader, val_loader, config: RouteConfig):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    train_maes, val_maes = [], []
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = general_regression_loss(model(xb), yb, config)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        train_mae = ep_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += general_regression_loss(model(xb), yb, config).item() * len(xb)
        val_mae = val_loss / len(val_loader.dataset)

        train_maes.append(train_mae)
        val_maes.append(val_mae)
        scheduler.step()

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = "  ← best"
        else:
            no_improve += 1
            marker = ""

        print(f"  Epoch {epoch:3d}/{EPOCHS}  train_MAE={train_mae:.3f}  val_MAE={val_mae:.3f}{marker}")

        if no_improve >= PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return train_maes, val_maes


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


def save_plots(train_maes, val_maes, y_true, y_pred, out_dir, model_name):
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_maes, label="Train MAE")
    ax.plot(val_maes, label="Val MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (bpm)")
    ax.set_title(f"{model_name} — Training Curves")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "train_metrics.png"), dpi=150)
    plt.close(fig)

    errors = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    lo = min(y_true.min(), y_pred.min()) - 5
    hi = max(y_true.max(), y_pred.max()) + 5
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=18)
    ax.plot([lo, hi], [lo, hi], "r--", label="Ideal")
    ax.set_xlabel("True HR (bpm)")
    ax.set_ylabel("Predicted HR (bpm)")
    ax.set_title(f"{model_name} — True vs Predicted")
    ax.text(0.05, 0.95, f"MAE  = {mae:.2f} bpm\nRMSE = {rmse:.2f} bpm", transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", alpha=0.3))
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "true_vs_predicted.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=40, edgecolor="k", alpha=0.75)
    ax.axvline(0, color="r", linestyle="--", label="Zero error")
    ax.set_xlabel("Prediction Error (bpm)")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} — Error Distribution")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_distribution.png"), dpi=150)
    plt.close(fig)

    idx = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, y_true, label="True HR", alpha=0.7)
    ax.plot(idx, y_pred, label="Predicted HR", alpha=0.7)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("HR (bpm)")
    ax.set_title(f"{model_name} — Predictions per Sample")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "prediction_vs_sample.png"), dpi=150)
    plt.close(fig)

    return mae, rmse


def save_train_test_comparison(train_mae, train_rmse, test_mae, test_rmse, out_dir, model_name):
    labels = ["Train", "Test"]
    maes = [train_mae, test_mae]
    rmses = [train_rmse, test_rmse]
    x = np.arange(2)
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, vals, metric in zip(axes, [maes, rmses], ["MAE (bpm)", "RMSE (bpm)"]):
        bars = ax.bar(x, vals, width * 2, color=["#4CAF50", "#F44336"], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{model_name} — {metric}", fontsize=11)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{value:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ratio = vals[1] / (vals[0] + 1e-12)
        ax.text(0.97, 0.95, f"x{ratio:.1f} ratio", transform=ax.transAxes, ha="right", va="top", fontsize=10, color="gray")

    plt.suptitle(f"{model_name} — Train vs Test Comparison\nTrain MAE={train_mae:.2f}  Test MAE={test_mae:.2f}  Train RMSE={train_rmse:.2f}  Test RMSE={test_rmse:.2f}", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_test_comparison.png"), dpi=150)
    plt.close(fig)


def save_metrics_summary(train_mae, train_rmse, test_mae, test_rmse, out_dir):
    pd.DataFrame([
        {"split": "train", "mae_bpm": train_mae, "rmse_bpm": train_rmse},
        {"split": "test", "mae_bpm": test_mae, "rmse_bpm": test_rmse},
    ]).to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)


def save_predictions_csv(test_names, y_true, y_pred, out_dir, file_name):
    pd.DataFrame({
        "file": test_names,
        "true_hr": y_true,
        "pred_hr": y_pred,
        "error_bpm": y_pred - y_true,
    }).to_csv(os.path.join(out_dir, file_name), index=False)


def run_experiment(
    config: RouteConfig,
    out_dir,
    prediction_file_name,
    training_seed=RANDOM_STATE,
    split_seed=RANDOM_STATE,
):
    set_seed(training_seed)

    print("=" * 72)
    print(config.model_name)
    print("=" * 72)
    print(f"  Training seed: {training_seed}")
    print(f"  Fixed split seed: {split_seed}")

    train_desc = "\nLoading & preprocessing training data ..." if config.use_full_pipeline else "\nLoading training data ..."
    test_desc = "Loading & preprocessing test data ..." if config.use_full_pipeline else "Loading test data ..."
    print(train_desc)
    X_all_raw, y_all, _ = load_folder(TRAIN_FOLDER, use_full_pipeline=config.use_full_pipeline)
    print(f"  {len(X_all_raw)} samples  shape {X_all_raw.shape}")

    print(test_desc)
    X_test_raw, y_test, test_names = load_folder(TEST_FOLDER, use_full_pipeline=config.use_full_pipeline)
    print(f"  {len(X_test_raw)} samples")

    print(f"  Computing inherited raw-robust spectral inputs (<= {config.feature_high:.1f} Hz) ...")
    X_all, freq_axis = compute_inherited_features(X_all_raw, config.feature_high)
    X_test, _ = compute_inherited_features(X_test_raw, config.feature_high)
    print(f"  Feature shape: {X_all.shape}  ({len(freq_axis)} freq bins, {X_all.shape[2]} channels)")

    train_idx, val_idx = make_train_val_split_indices(y_all, split_seed=split_seed, val_size=0.2)
    X_tr, X_val = X_all[train_idx], X_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]
    print("  Using standard general training without HR-band weighting or stratification")

    mean, std = fit_scaler(X_tr)
    X_tr = apply_scaler(X_tr, mean, std)
    X_val = apply_scaler(X_val, mean, std)
    X_te = apply_scaler(X_test, mean, std)

    train_loader = make_loader(X_tr, y_tr, shuffle=True, augment=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_te, y_test, shuffle=False)

    model = InheritedRawRobustCNN(freq_axis=freq_axis, config=config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    print("\nTraining ...")
    train_maes, val_maes = train_model(model, train_loader, val_loader, config)

    print("\nEvaluating ...")
    y_pred = predict(model, test_loader)
    tr_eval_loader = make_loader(X_tr, y_tr, shuffle=False)
    y_tr_pred = predict(model, tr_eval_loader)
    train_mae = mean_absolute_error(y_tr, y_tr_pred)
    train_rmse = np.sqrt(mean_squared_error(y_tr, y_tr_pred))
    test_mae, test_rmse = save_plots(train_maes, val_maes, y_test, y_pred, out_dir, config.model_name)

    print("\n" + "=" * 40)
    print(f"  Train MAE  : {train_mae:.3f} bpm")
    print(f"  Train RMSE : {train_rmse:.3f} bpm")
    print(f"  Test  MAE  : {test_mae:.3f} bpm")
    print(f"  Test  RMSE : {test_rmse:.3f} bpm")
    print("=" * 40)

    save_train_test_comparison(train_mae, train_rmse, test_mae, test_rmse, out_dir, config.model_name)
    save_metrics_summary(train_mae, train_rmse, test_mae, test_rmse, out_dir)
    save_predictions_csv(test_names, y_test, y_pred, out_dir, prediction_file_name)

    print(f"\nAll outputs saved to:\n  {out_dir}")
    return {
        "training_seed": int(training_seed),
        "split_seed": int(split_seed),
        "train_mae": float(train_mae),
        "train_rmse": float(train_rmse),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
    }