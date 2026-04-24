"""
regression.py
Trains a Random Forest Regressor to predict any numeric column
from the remaining columns in an uploaded CSV.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")


def preprocess_for_ml(df: pd.DataFrame, target_col: str):
    """
    Prepare a DataFrame for ML:
    - Drop rows where target is null
    - Encode categoricals
    - Fill remaining nulls with median/mode
    Returns X, y, feature_names, encoders
    """
    df = df.copy()
    df = df.dropna(subset=[target_col])

    y = df[target_col].values
    X = df.drop(columns=[target_col])

    encoders = {}
    for col in X.columns:
        # Try converting to numeric first
        converted = pd.to_numeric(X[col], errors='coerce')
        if converted.notna().sum() / max(len(converted), 1) >= 0.5:
            X[col] = converted

        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = X[col].astype(str).fillna("missing")
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())

    

    feature_names = X.columns.tolist()
    return X.values, y, feature_names, encoders


def train_regression_model(df: pd.DataFrame, target_col: str) -> dict:
    """
    Train a RandomForestRegressor to predict target_col.
    Returns metrics, feature importances, and saves the model.
    """
    X, y, feature_names, encoders = preprocess_for_ml(df, target_col)

    if len(X) < 10:
        return {"error": "Not enough rows to train (minimum 10 required)."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Feature importances
    importances = dict(zip(feature_names, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"regression_{target_col}.pkl")
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "encoders": encoders,
        "target_col": target_col,
    }, model_path)

    return {
        "model_type": "Random Forest Regressor",
        "target": target_col,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "feature_importances": importances,
        "model_path": model_path,
    }


def predict_regression(model_path: str, input_data: dict) -> float:
    """
    Load a saved regression model and predict a value.
    input_data: dict of {feature_name: value}
    """
    saved = joblib.load(model_path)
    model = saved["model"]
    feature_names = saved["feature_names"]
    encoders = saved["encoders"]

    row = []
    for col in feature_names:
        val = input_data.get(col, 0)
        if col in encoders:
            try:
                val = encoders[col].transform([str(val)])[0]
            except ValueError:
                val = 0
        row.append(val)

    return float(model.predict([row])[0])