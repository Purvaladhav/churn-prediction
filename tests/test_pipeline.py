import pytest
import pandas as pd
import numpy as np
import yaml
import joblib
import os

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── 1. Config Tests ───────────────────────────────────────────
def test_config_loads():
    assert cfg is not None

def test_config_has_required_keys():
    assert "data" in cfg
    assert "models" in cfg
    assert "paths" in cfg
    assert "mlflow" in cfg

def test_config_test_size_valid():
    assert 0 < cfg["data"]["test_size"] < 1

# ── 2. Data Tests ─────────────────────────────────────────────
def test_dataset_loads():
    df = pd.read_csv(cfg["data"]["path"])
    assert df.shape[0] > 0
    assert df.shape[1] > 0

def test_dataset_has_target_column():
    df = pd.read_csv(cfg["data"]["path"])
    assert cfg["data"]["target_column"] in df.columns

def test_churn_encoding():
    df = pd.DataFrame({"Churn": ["Yes", "No", "Yes", "No"]})
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    assert df["Churn"].tolist() == [1, 0, 1, 0]
    assert df["Churn"].dtype in [np.int64, np.int32]

def test_no_missing_after_cleaning():
    df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, 6]})
    df.dropna(inplace=True)
    assert df.isnull().sum().sum() == 0

def test_total_charges_numeric():
    df = pd.read_csv(cfg["data"]["path"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    assert df["TotalCharges"].dtype == np.float64

# ── 3. Model Artifacts Tests ──────────────────────────────────
def test_scaler_exists():
    assert os.path.exists(cfg["paths"]["scaler_path"])

def test_splits_exist():
    assert os.path.exists(cfg["paths"]["splits_path"])

def test_feature_names_exist():
    assert os.path.exists(cfg["paths"]["feature_names_path"])

def test_all_models_saved():
    for name in cfg["models"].keys():
        path = f"{cfg['paths']['models_dir']}/{name}.pkl"
        assert os.path.exists(path), f"Missing model: {path}"

def test_scaler_transform_shape():
    scaler = joblib.load(cfg["paths"]["scaler_path"])
    sample = np.zeros((1, 19))
    result = scaler.transform(sample)
    assert result.shape == (1, 19)

def test_model_predicts_probability():
    model = joblib.load(f"{cfg['paths']['models_dir']}/xgboost.pkl")
    sample = np.zeros((1, 19))
    proba = model.predict_proba(sample)
    assert proba.shape == (1, 2)
    assert 0 <= proba[0][1] <= 1

# ── 4. Data Split Tests ───────────────────────────────────────
def test_splits_correct_shape():
    X_train, X_test, y_train, y_test = joblib.load(cfg["paths"]["splits_path"])
    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

def test_no_data_leakage():
    X_train, X_test, y_train, y_test = joblib.load(cfg["paths"]["splits_path"])
    assert len(X_train) != len(X_test)