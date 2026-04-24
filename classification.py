"""
classification.py
Trains a Random Forest Classifier to predict any categorical column
from the remaining columns in an uploaded CSV.
"""

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")


def preprocess_for_classification(df: pd.DataFrame, target_col: str):
    """
    Prepare DataFrame for classification:
    - Encode target column
    - Encode all categorical features
    - Fill nulls
    Returns X, y, feature_names, feature_encoders, target_encoder
    """
    df = df.copy()
    df = df.dropna(subset=[target_col])

    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[target_col].astype(str))

    X = df.drop(columns=[target_col])
    feature_encoders = {}

    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col])
            feature_encoders[col] = le
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X.values, y, X.columns.tolist(), feature_encoders, target_encoder


def train_classification_model(df: pd.DataFrame, target_col: str) -> dict:
    """
    Train a RandomForestClassifier to predict target_col.
    Returns accuracy, report, feature importances, and saves the model.
    """
    # Guard check: prevent classifying on continuous numeric data (like Salary)
    if is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20:
        return {"error": f"Column '{target_col}' has too many unique numeric values ({df[target_col].nunique()}). This is a continuous variable. Please use the Regression tab instead."}

    X, y, feature_names, feature_encoders, target_encoder = preprocess_for_classification(
        df, target_col
    )

    if len(X) < 10:
        return {"error": "Not enough rows to train (minimum 10 required)."}

    class_counts = np.bincount(y)
    if len(class_counts) < 2:
        return {"error": "Target column must have at least 2 unique classes."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if min(class_counts) >= 2 else None
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    class_names = target_encoder.classes_.tolist()

    # Only use labels present in test split — avoids size mismatch error
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names = [class_names[i] for i in present_labels if i < len(class_names)]

    report = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=present_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix aligned to same label set
    cm = confusion_matrix(y_test, y_pred, labels=present_labels).tolist()
    # Update classes to only present ones for display
    class_names = present_names

    # Feature importances
    importances = dict(zip(feature_names, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"classification_{target_col}.pkl")
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "feature_encoders": feature_encoders,
        "target_encoder": target_encoder,
        "target_col": target_col,
        "class_names": class_names,
    }, model_path)

    return {
        "model_type": "Random Forest Classifier",
        "target": target_col,
        "classes": class_names,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importances": importances,
        "model_path": model_path,
    }


def predict_classification(model_path: str, input_data: dict) -> dict:
    """
    Load a saved classification model and predict a class.
    Returns predicted class and probability for each class.
    """
    saved = joblib.load(model_path)
    model = saved["model"]
    feature_names = saved["feature_names"]
    feature_encoders = saved["feature_encoders"]
    target_encoder = saved["target_encoder"]
    class_names = saved["class_names"]

    row = []
    for col in feature_names:
        val = input_data.get(col, 0)
        if col in feature_encoders:
            try:
                val = feature_encoders[col].transform([str(val)])[0]
            except ValueError:
                val = 0
        else:
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
        row.append(val)

    pred_enc = model.predict([row])[0]
    probas = model.predict_proba([row])[0]

    return {
        "predicted_class": target_encoder.inverse_transform([pred_enc])[0],
        "probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(class_names, probas)
        },
    }