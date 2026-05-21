"""Tasks 7-12: split → scale → train LR & RF → evaluate → summary table"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
RES_DIR = "results"
FIG_DIR = "figures"
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load features ─────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(RES_DIR, "features.csv"))
feature_cols = [c for c in df.columns if c not in ("file", "HR")]
X = df[feature_cols].values
y = df["HR"].values
print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features")

# ── Task 7: 80/20 split ───────────────────────────────────────────────────────
print("\n=== Task 7: Train/Test Split (80/20) ===")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train: {X_train_raw.shape[0]} samples  |  Test: {X_test_raw.shape[0]} samples")

# Save raw split with labels
train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42
)
df.iloc[train_idx].to_csv(os.path.join(RES_DIR, "train_raw.csv"), index=False)
df.iloc[test_idx].to_csv(os.path.join(RES_DIR, "test_raw.csv"),  index=False)
print("  Saved: results/train_raw.csv, results/test_raw.csv")

# ── Task 8: Feature scaling (fit on train only) ───────────────────────────────
print("\n=== Task 8: Feature Scaling (StandardScaler) ===")
scaler    = StandardScaler()
X_train   = scaler.fit_transform(X_train_raw)
X_test    = scaler.transform(X_test_raw)

pd.DataFrame(X_train, columns=feature_cols).assign(HR=y_train).to_csv(
    os.path.join(RES_DIR, "X_train.csv"), index=False)
pd.DataFrame(X_test,  columns=feature_cols).assign(HR=y_test).to_csv(
    os.path.join(RES_DIR, "X_test.csv"),  index=False)
print("  Scaler fit on train only — no data leakage.")
print("  Saved: results/X_train.csv, results/X_test.csv")

# ── Helper: evaluate ─────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {name:30s}  MAE={mae:.4f} bpm   RMSE={rmse:.4f} bpm")
    return mae, rmse

# ── Task 9: Linear Regression ─────────────────────────────────────────────────
print("\n=== Task 9: Linear Regression ===")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr, rmse_lr = evaluate("Linear Regression", y_test, y_pred_lr)

pd.DataFrame({"HR_true": y_test, "HR_pred_LR": y_pred_lr}).to_csv(
    os.path.join(RES_DIR, "lr_predictions.csv"), index=False)
print("  Saved: results/lr_predictions.csv")

# ── Task 10: Random Forest ────────────────────────────────────────────────────
print("\n=== Task 10: Random Forest ===")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf, rmse_rf = evaluate("Random Forest", y_test, y_pred_rf)

pd.DataFrame({"HR_true": y_test, "HR_pred_RF": y_pred_rf}).to_csv(
    os.path.join(RES_DIR, "rf_predictions.csv"), index=False)
print("  Saved: results/rf_predictions.csv")

# Feature importance plot
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 7))
importances.plot(kind="barh", ax=ax, color="#4C72B0")
# ax.set_xlabel("Feature Importance", fontsize=12)  # Removed for academic thesis formatting compliance
# ax.set_title("Random Forest — Feature Importances", fontsize=13)  # Removed for academic thesis formatting compliance
sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "rf_feature_importance.png"), dpi=150)
plt.close()
print("  Saved: figures/rf_feature_importance.png")

# ── Task 11: Scatter plots ────────────────────────────────────────────────────
print("\n=== Task 11: Evaluation Plots ===")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_pred, title, mae, rmse in zip(
        axes,
        [y_pred_lr, y_pred_rf],
        ["Linear Regression", "Random Forest"],
        [mae_lr, mae_rf],
        [rmse_lr, rmse_rf]):
    ax.scatter(y_test, y_pred, alpha=0.4, s=18, color="#4C72B0")
    lims = [min(y_test.min(), y_pred.min()) - 3,
            max(y_test.max(), y_pred.max()) + 3]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal (y=x)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True HR (bpm)", fontsize=12)
    ax.set_ylabel("Predicted HR (bpm)", fontsize=12)
    ax.set_title(f"{title}\nMAE={mae:.2f} bpm  RMSE={rmse:.2f} bpm", fontsize=12)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "ml_model_scatter.png"), dpi=150)
plt.close()
print("  Saved: figures/ml_model_scatter.png")

# ── Task 12: Model Comparison Summary Table ───────────────────────────────────
print("\n=== Task 12: Model Comparison Summary Table ===")

# Load peak-based results from Task 4
peak_raw  = pd.read_csv(os.path.join(RES_DIR, "peak_hr_estimates_raw.csv")).dropna()
peak_norm = pd.read_csv(os.path.join(RES_DIR, "peak_hr_estimates_norm.csv")).dropna()
mae_peak_raw  = mean_absolute_error(peak_raw["HR_true"],  peak_raw["HR_estimated"])
rmse_peak_raw = np.sqrt(mean_squared_error(peak_raw["HR_true"],  peak_raw["HR_estimated"]))
mae_peak_norm  = mean_absolute_error(peak_norm["HR_true"], peak_norm["HR_estimated"])
rmse_peak_norm = np.sqrt(mean_squared_error(peak_norm["HR_true"], peak_norm["HR_estimated"]))

summary = pd.DataFrame([
    {"Model": "Peak-based (no normalisation)",   "MAE_bpm": round(mae_peak_raw,  2), "RMSE_bpm": round(rmse_peak_raw,  2), "Notes": "Baseline, no ML"},
    {"Model": "Peak-based (with normalisation)", "MAE_bpm": round(mae_peak_norm, 2), "RMSE_bpm": round(rmse_peak_norm, 2), "Notes": "Baseline, no ML"},
    {"Model": "Linear Regression",               "MAE_bpm": round(mae_lr,  2),       "RMSE_bpm": round(rmse_lr,  2),       "Notes": "18 features, 80/20 split"},
    {"Model": "Random Forest (100 trees)",       "MAE_bpm": round(mae_rf,  2),       "RMSE_bpm": round(rmse_rf,  2),       "Notes": "18 features, 80/20 split"},
])

summary.to_csv(os.path.join(RES_DIR, "model_summary.csv"), index=False)

print("\n" + "=" * 65)
print(f"{'Model':<35} {'MAE (bpm)':>10} {'RMSE (bpm)':>11}")
print("-" * 65)
for _, row in summary.iterrows():
    print(f"  {row['Model']:<33} {row['MAE_bpm']:>10.2f} {row['RMSE_bpm']:>11.2f}")
print("=" * 65)
print("\nSaved: results/model_summary.csv")
print("\nAll tasks complete.")
