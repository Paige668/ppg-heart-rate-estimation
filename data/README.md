# Data Directory Structure

To reproduce the experiments in this repository, you need to download the public dataset and place the CSV files in the designated folders.

## 📥 Public Dataset Download Guide (Dataset 1)

This project utilizes the **IEEE Signal Processing Cup 2015 (TROIKA)** dataset for training and testing.

1. **Download link:** [Zenodo TROIKA Dataset (12-subject structured)](https://zenodo.org/records/3902710)
2. **Download files:**
   * **Training data:** Download the training fold files and place all training `.csv` files inside the `data/samples_train_csv/` folder.
   * **Test data:** Download the test fold files and place all test `.csv` files inside the `data/samples_test_csv/` folder.
3. **Resting-state data:**
   * For the static resting-state feature analysis (RQ1), place the 10-second sliding windows inside the `data/ppg_10s_windows_rest/` folder.

## 📁 Expected Directory Structure

After placing the datasets, your `data/` directory must look exactly like this:

```
ppg_heartrate_ml/
└── data/
    ├── README.md                          # This file
    ├── ppg_10s_windows_rest/              # Static PPG 10s windows (.csv)
    │   ├── sample_rest_1.csv
    │   └── ...
    ├── samples_train_csv/                 # Dynamic training samples (.csv)
    │   ├── sample_1_HR_72.csv
    │   └── ...
    └── samples_test_csv/                  # Dynamic test samples (.csv)
        ├── sample_1039_HR_100.csv
        └── ...
```

---

## 🚫 Private Dataset Exemption (Dataset 2)

**Dataset 2 (Kalmar study)** contains private patient health recordings collected under institutional review board approval (IRB DNR 2023-04335-01). 
* Due to strict medical ethical constraints and GDPR requirements, these raw datasets **cannot be shared publicly**.
* The thesis results and code reproducibility are validated using the public **Dataset 1 (TROIKA)** as the primary open-source benchmark.
