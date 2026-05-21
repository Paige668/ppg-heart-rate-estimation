# Heart Rate Prediction from PPG Signals Using Machine Learning Approaches

This repository contains the official implementation of the signal preprocessing pipeline and machine learning models for the thesis:
**"Heart Rate Prediction from Photoplethysmography Signals Using Machine Learning Approaches"**.

This project provides a fully reproducible pipeline starting from raw photoplethysmography (PPG) and tri-axial accelerometer (ACC) signals, progressing through Singular Spectrum Analysis (SSA) and Joint Sparse Spectrum (JOSS) preprocessing, to heart rate estimation using traditional ML models and a Unified Spectral Convolutional Neural Network (CNN).

---

## Repository Structure

The code is modularly organized to correspond directly with the Research Questions (RQs) and experimental sections of the thesis:

```bash
ppg_heartrate_ml/
├── README.md                          # This file (Complete run guide & results lookup)
├── requirements.txt                   # Python environment dependencies
├── .gitignore                         # Exclude cache, environments, and datasets
│
├── data/                              # Dataset folders
│   └── README.md                      # Zenodo public dataset download & placement instructions
│
├── rq1_resting_state/                 # RQ1: Resting-State Feature-Based Analysis
│   ├── rq1_cross_validation.py        # 5-fold cross-validation for traditional models (Table 5.1)
│   ├── rf_pipeline.py                 # Independent RF pipeline
│   ├── extract_resting_features.py    # Baseline feature extraction (20 handcrafted features)
│   └── train_rf_and_plot_importance.py # Generates RF feature importance plot (Figure 5.1)
│
├── rq2_pipeline_vs_baseline/          # RQ2: Signal Preprocessing Pipeline Evaluation
│   ├── baseline_algorithm.py          # Peak-based tracking benchmark (Table 5.2 baseline row)
│   └── ppg_pipeline.py                # Full SSA+JOSS pipeline (Table 5.2 pipeline, Figure 5.3)
│
└── rq2_dynamic_models/                # RQ2: Dynamic Motion-Active Model Benchmark
    ├── shared/                        # Core shared modules
    │   ├── feature_engine.py          # 32 matched feature extraction (20 PPG + 12 ACC)
    │   ├── prep_engine.py             # SSA + Bandpass signal cleaning module
    │   └── train_engine.py            # Generic dataset loader and metric visualization
    │
    ├── cnn/                           # Unified Convolutional Neural Network
    │   ├── cnn_unified.py             # Main CNN script (No-Prep & Full Pipeline configurations)
    │   └── run_cnn_stability.py       # 5-seed stability sweep for CNN (Table 5.3 CNN row)
    │
    ├── random_forest/                 # Strict-matched Random Forest Model
    │   ├── run_rf_matched.py          # Main RF runner
    │   └── run_rf_stability.py        # 5-seed stability sweep for RF (Table 5.3 RF row)
    │
    ├── xgboost/                       # Strict-matched XGBoost Model
    │   ├── run_xgb_matched.py         # Main XGBoost runner
    │   └── run_xgb_stability.py       # 5-seed stability sweep for XGBoost (Table 5.3 XGB row)
    │
    └── comparison/                    # Thesis Comparison Results Generator
        └── generate_comparison_results.py # Aggregates all model CSVs and builds charts (Figures 5.4 & 5.5)
```

---

## Setup & Installation

### Step 1: Clone and Set Up Virtual Environment

Initialize a clean Python environment (Python 3.8+ recommended):

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Download the Datasets

Refer to the instructions inside [data/README.md](data/README.md) to download **Dataset 1 (TROIKA)** from Zenodo and place the CSV files into:

* `data/ppg_10s_windows_rest/` (Static rest windows)
* `data/samples_train_csv/` (Dynamic train fold)
* `data/samples_test_csv/` (Dynamic test fold)

---

## Thesis Results Reproduction Guide

Follow the instructions below to run each experimental module and verify it matches the results documented in Chapter 5.

### RQ1: Resting-State Feature Models (Table 5.1 & Figure 5.1)

Run the 5-fold cross-validation script to generate the mean ± standard deviation metrics:

```bash
python rq1_resting_state/rq1_cross_validation.py
```

* **Expected Output:** Terminal output and `results/rq1_cv_summary.csv` will show cross-validation metrics.
* **Feature Importance:** To extract features and save the 20 relative feature importances shown in **Figure 5.1**, run:

```bash
python rq1_resting_state/extract_resting_features.py
python rq1_resting_state/train_rf_and_plot_importance.py
```

* The chart is saved to `rq1_resting_state/figures/rf_feature_importance.png`.

#### **Table 5.1 Verification Lookup Table (RQ1 CV Results)**

| Model | Expected CV MAE (bpm) | Expected CV RMSE (bpm) |
| --- | --- | --- |
| **Linear Regression** | $11.16 \pm 0.44$ | $14.47 \pm 0.55$ |
| **K-Nearest Neighbors (K=10)** | $7.44 \pm 0.32$ | $10.45 \pm 0.41$ |
| **Random Forest (100 trees)** | $6.23 \pm 0.28$ | $8.91 \pm 0.35$ |

---

### RQ2: Preprocessing (SSA+JOSS) Pipeline vs. Peak-Detection Baseline (Table 5.2 & Figure 5.3)

Run the peak-detection baseline algorithm and the proposed preprocessing pipeline sequentially on the test dataset:

```bash
# 1. Run Peak-detection baseline
python rq2_pipeline_vs_baseline/baseline_algorithm.py

# 2. Run proposed SSA+JOSS pipeline
python rq2_pipeline_vs_baseline/ppg_pipeline.py
```

* **Expected Output:**
  * Baseline overall stats: Terminal printout.
  * Preprocessing Pipeline: Generates `rq2_pipeline_vs_baseline/output_plots/07_pipeline_test_summary.png` which maps MAE and RMSE performance across the 4 heart rate intensity groups (matching **Figure 5.3**).

#### **Table 5.2 Verification Lookup Table (Pipeline vs Baseline)**

| Preprocessing Configuration | Test MAE (bpm) | Test RMSE (bpm) |
| --- | --- | --- |
| **No-Prep Baseline (Peak-Based)** | $23.86$ | $33.86$ |
| **Proposed Pipeline (SSA + JOSS)** | $3.47$ | $4.87$ |

---

### RQ2: Dynamic Multi-Model Comparison (Table 5.3, Figures 5.4 & 5.5)

To evaluate Random Forest, XGBoost, and the Unified Convolutional Neural Network under **No-Prep** (raw) and **Full Pipeline** (cleaned) configurations across **5 random training seeds**:

```bash
# 1. Run Random Forest canonical matching & stability sweeps
python rq2_dynamic_models/random_forest/run_rf_matched.py
python rq2_dynamic_models/random_forest/run_rf_stability.py

# 2. Run XGBoost canonical matching & stability sweeps
python rq2_dynamic_models/xgboost/run_xgb_matched.py
python rq2_dynamic_models/xgboost/run_xgb_stability.py

# 3. Run Unified CNN canonical matching & stability sweeps
python rq2_dynamic_models/cnn/cnn_unified.py
python rq2_dynamic_models/cnn/run_cnn_stability.py
```

After all three model sweeps complete and output their predictions to their respective directories, run the **Results Comparison Generator** to compile the thesis figures and group metrics:

```bash
python rq2_dynamic_models/comparison/generate_comparison_results.py
```

* **Expected Output:**
  * **Table 5.3 Data:** Saved to `rq2_dynamic_models/comparison_results/stability_summary.csv`.
  * **Figure 5.4 Chart:** Saved to `rq2_dynamic_models/comparison_results/charts/canonical_test_combined.png` showing combined MAE and RMSE histograms side-by-side.
  * **Figure 5.5 CNN Intensity Breakdown:** Saved to `rq2_dynamic_models/comparison_results/charts/cnn_no_prep_hr_group_mae.png`.

#### **Table 5.3 Verification Lookup Table (5-Seed Stability Results)**

| Model Family | Preprocessing | Test MAE (mean ± std bpm) | Test RMSE (mean ± std bpm) |
| --- | --- | --- | --- |
| **Random Forest** | No-Prep | $16.48 \pm 0.08$ | $22.42 \pm 0.09$ |
| **Random Forest** | Full Pipeline | $13.88 \pm 0.06$ | $19.42 \pm 0.07$ |
| **XGBoost** | No-Prep | $12.30 \pm 0.05$ | $17.50 \pm 0.06$ |
| **XGBoost** | Full Pipeline | $10.50 \pm 0.04$ | $15.10 \pm 0.05$ |
| **Unified CNN** | No-Prep | **$4.20 \pm 0.15$** | **$6.10 \pm 0.18$** |
| **Unified CNN** | Full Pipeline | **$3.80 \pm 0.12$** | **$5.50 \pm 0.15$** |

---

## Reproducibility and Calibration Notes

1. **CPU/GPU Consistency:** All deep learning loops (CNN) are set to run on `CPU` (`DEVICE = torch.device("cpu")`) to prevent CUDA tensor alignment variances on different hardware.
2. **Fixed Random Splits:** All train/validation splits in the dynamic scripts are hardcoded to `split_seed=42` to guarantee that the models are consistently evaluated on the exact same unseen test samples.
3. **Collinearity Protection:** Tree models utilize standard `StandardScaler` fitted *strictly* on training indices to prevent information leakage.
