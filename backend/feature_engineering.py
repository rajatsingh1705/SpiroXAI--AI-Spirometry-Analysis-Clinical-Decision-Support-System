"""
Feature engineering for spirometry prediction.
Replicates the exact feature engineering from the training notebook (h200.ipynb).
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "saved_models")


def add_spirometry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to match the training pipeline exactly.
    This must replicate Cell 10 of h200.ipynb feature engineering.
    """
    df = df.copy()
    existing = set(df.columns)

    def add(name, val):
        if name not in existing:
            df[name] = val

    pef  = "Baseline_PEF_Ls"
    fef  = "Baseline_FEF2575_Ls"
    evol = "Baseline_Extrapolated_Volume"
    fet  = "Baseline_Forced_Expiratory_Time"

    # Log transforms
    for col in [pef, fef, evol, fet]:
        if col in existing:
            add(f"log_{col}", np.log1p(df[col].clip(lower=1e-6)))

    # Ratio and cross features
    if pef in existing and fef in existing:
        add("FEF_PEF_Ratio", df[fef] / (df[pef] + 1e-9))
        add("PEF_x_FEF",     df[pef] * df[fef])
        add("PEF_minus_FEF", df[pef] - df[fef])

    if evol in existing and fet in existing:
        add("Vol_per_FET", df[evol] / (df[fet] + 1e-9))

    if "Age" in existing:
        if pef in existing: add("PEF_per_Age", df[pef] / (df["Age"] + 1e-9))
        if fef in existing: add("FEF_per_Age", df[fef] / (df["Age"] + 1e-9))

    if "Height" in existing:
        add("Height_sq", df["Height"] ** 2)
        if pef in existing: add("PEF_per_Ht", df[pef] / (df["Height"] + 1e-9))
        if fef in existing: add("FEF_per_Ht", df[fef] / (df["Height"] + 1e-9))

    if "BMI" in existing:
        if pef in existing: add("BMI_x_PEF", df["BMI"] * df[pef])
        if fef in existing: add("BMI_x_FEF", df["BMI"] * df[fef])

    if "Sex" in existing:
        if pef in existing: add("Sex_x_PEF", df["Sex"] * df[pef])
        if fef in existing: add("Sex_x_FEF", df["Sex"] * df[fef])

    if pef in existing: add("PEF_sq", df[pef] ** 2)
    if fef in existing: add("FEF_sq", df[fef] ** 2)

    return df


def load_quantile_transformer():
    """Load the QuantileTransformer scaler from saved_models/qt_full.pkl."""
    qt_path = os.path.join(MODELS_DIR, "qt_full.pkl")
    if os.path.exists(qt_path):
        with open(qt_path, "rb") as f:
            return pickle.load(f)
    return None


def load_feature_names():
    """Load feature names list. Tries pickle first, falls back to experiment_summary.json."""
    pkl_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    summary_path = os.path.join(PROJECT_DIR, "experiment_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
            return data.get("feature_names", [])

    return []


def prepare_features(raw_input: dict, feature_names: list, qt=None):
    """
    Apply feature engineering and return (X_raw, X_scaled).

    X_raw:    numpy array (1, n_features) — unscaled, for XGBoost/LightGBM
    X_scaled: numpy array (1, n_features) — QT-scaled, for DNN/FT-Transformer
    """
    df = pd.DataFrame([raw_input])
    df = add_spirometry_features(df)

    # Reindex to exact feature order, fill missing with 0
    X = df.reindex(columns=feature_names, fill_value=0).values.astype(np.float32)

    if qt is not None:
        X_scaled = qt.transform(X).astype(np.float32)
    else:
        X_scaled = X.copy()

    return X, X_scaled
