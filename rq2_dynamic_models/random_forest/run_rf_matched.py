import os
import sys
import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Setup shared paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.abspath(os.path.join(_HERE, "../shared"))
if _SHARED not in sys.path:
    sys.path.append(_SHARED)

import feature_engine
import prep_engine

def find_workspace_root(start_dir):
    current = start_dir
    while True:
        if os.path.isdir(os.path.join(current, "data")):
            return current
        parent = os.path.dirname(current)
        if parent == current: return None
        current = parent

_ROOT = find_workspace_root(_HERE)
TRAIN_FOLDER = os.path.join(_ROOT, "data", "samples_train_csv")
TEST_FOLDER  = os.path.join(_ROOT, "data", "samples_test_csv")

def _sort_key(p):
    try: return int(os.path.basename(p).split("_")[1])
    except: return 0

def load_matched_data(folder, use_full_pipeline=False):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")), key=_sort_key)
    print(f"Loading {len(files)} files from {os.path.basename(folder)} (Full Pipeline: {use_full_pipeline})")
    all_feats, all_hrs = [], []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        # RF General Baseline Cols: dim0=PPG, dim2=AccX, dim3=AccY, dim4=AccZ (Checking original file)
        # Wait, let's verify columns from previous read_file. 
        # PPG_COL="dim0", ACCX_COL="dim2", ACCY_COL="dim3", ACCZ_COL="dim4"
        ppg = df["dim0"].values
        ax = df["dim2"].values
        ay = df["dim3"].values
        az = df["dim4"].values
        hr = df["label"].iloc[0]
        
        # Preprocess PPG if needed
        ppg_proc = prep_engine.preprocess_signal(ppg, ax, ay, az, use_full_pipeline=use_full_pipeline)
        
        # Extract 32 features (20 PPG + 12 Acc)
        feats = feature_engine.extract_matched_features(ppg_proc, ax, ay, az)
        all_feats.append(feats)
        all_hrs.append(hr)
        if (i+1) % 1000 == 0: print(f"  {i+1}/{len(files)}...")
        
    df_x = pd.DataFrame(all_feats)
    df_x = df_x.fillna(df_x.median()) # Basic robustness
    return df_x, np.array(all_hrs)

def build_model(training_seed):
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        random_state=training_seed,
        n_jobs=-1,
    )


def run_rf_experiment(
    use_full_pipeline,
    case_name,
    training_seed=42,
    output_dir=None,
    prediction_file_name=None,
):
    print(f"\n--- Running RF Strict Matched: {case_name} ---")
    X_train, y_train = load_matched_data(TRAIN_FOLDER, use_full_pipeline=use_full_pipeline)
    X_test, y_test = load_matched_data(TEST_FOLDER, use_full_pipeline=use_full_pipeline)

    rf = build_model(training_seed)
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    y_pred = rf.predict(X_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(
        f"[{case_name}] seed={training_seed} Results: "
        f"Train MAE={train_mae:.4f}, Train RMSE={train_rmse:.4f}, "
        f"Test MAE={mae:.4f}, Test RMSE={rmse:.4f}"
    )

    # Save predictions
    if output_dir is None:
        output_dir = _HERE
    os.makedirs(output_dir, exist_ok=True)
    if prediction_file_name is None:
        prediction_file_name = f"rf_strict_matched_{case_name}_predictions.csv"
    out_csv = os.path.join(output_dir, prediction_file_name)
    pd.DataFrame({"true": y_test, "pred": y_pred}).to_csv(out_csv, index=False)

    return {
        "case": case_name,
        "training_seed": training_seed,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "test_mae": mae,
        "test_rmse": rmse,
        "mae": mae,
        "rmse": rmse,
    }

if __name__ == "__main__":
    results = []
    results.append(run_rf_experiment(False, "no_prep"))
    results.append(run_rf_experiment(True, "full_pipeline"))
    
    import json
    with open(os.path.join(_HERE, "rf_strict_matched_summary.json"), "w") as f:
        json.dump(results, indent=2, fp=f)
