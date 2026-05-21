import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys

# Add shared path to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
import feature_engine

def find_workspace_root(start_dir):
    current = start_dir
    while True:
        if os.path.isdir(os.path.join(current, "data")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError("Root not found")
        current = parent

_ROOT = find_workspace_root(_HERE)
TRAIN_FOLDER = os.path.join(_ROOT, "data", "samples_train_csv")
TEST_FOLDER  = os.path.join(_ROOT, "data", "samples_test_csv")

def _sort_key(path):
    try: return int(os.path.basename(path).split("_")[1])
    except: return 0

def load_data(folder, use_full_pipeline=False):
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")), key=_sort_key)
    records, hr_values = [], []
    
    # Optional preprocessing mock (SSA/JOSS logic would go here if we were actually processing)
    # For 'strict matched', the caller will provide the signal state
    
    for i, f in enumerate(csv_files):
        df = pd.read_csv(f)
        # Assuming cols: dim0 (PPG), dim1 (AccX), dim2 (AccY), dim3 (AccZ), label (HR)
        ppg = df['dim0'].values
        ax = df['dim1'].values
        ay = df['dim2'].values
        az = df['dim3'].values
        hr = float(df['label'].iloc[0])
        
        # If use_full_pipeline=True, we would normally call the cleaning C++ or Python logic here
        # But for this script, we assume the signals passed to feature_engine are already handled by the runner.
        
        feat = feature_engine.extract_matched_features(ppg, ax, ay, az)
        records.append(feat)
        hr_values.append(hr)
        if (i+1) % 500 == 0: print(f"  Processed {i+1}/{len(csv_files)}...")
        
    return pd.DataFrame(records), np.array(hr_values)

def plot_results(y_true, y_pred, title, out_path):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title(f"{title}\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")
    plt.xlabel("True HR")
    plt.ylabel("Pred HR")
    plt.savefig(out_path)
    plt.close()
    return mae, rmse
