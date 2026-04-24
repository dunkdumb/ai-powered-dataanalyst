"""
data_quality.py
Auto-detects column types and data quality issues from any CSV.
Uses rule-based heuristics + a trained RandomForest classifier.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/column_type_model.pkl")


# ── Feature extraction from a single column ───────────────────────────────────

def extract_column_features(series: pd.Series) -> dict:
    """Extract numeric features from a pandas Series for model input."""
    total = len(series)
    non_null = series.dropna()
    null_count = total - len(non_null)

    # Try numeric conversion
    numeric_converted = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = numeric_converted.notna().sum() / max(len(non_null), 1)

    # Unique ratio
    unique_ratio = series.nunique() / max(total, 1)

    # String length stats
    str_series = non_null.astype(str)
    avg_len = str_series.str.len().mean() if len(str_series) > 0 else 0
    max_len = str_series.str.len().max() if len(str_series) > 0 else 0

    # Date detection
    date_ratio = 0.0
    try:
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        date_ratio = parsed.notna().sum() / max(len(non_null), 1)
    except Exception:
        pass

    # Boolean-like detection
    bool_values = {"true", "false", "yes", "no", "0", "1", "t", "f", "y", "n"}
    bool_ratio = str_series.str.lower().isin(bool_values).sum() / max(len(str_series), 1)

    return {
        "null_ratio": null_count / max(total, 1),
        "numeric_ratio": float(numeric_ratio),
        "unique_ratio": float(unique_ratio),
        "avg_str_len": float(avg_len),
        "max_str_len": float(max_len),
        "date_ratio": float(date_ratio),
        "bool_ratio": float(bool_ratio),
        "total_rows": total,
    }


def detect_column_type(series: pd.Series) -> str:
    """Rule-based column type detection (used as labels for training)."""
    features = extract_column_features(series)

    if features["bool_ratio"] > 0.9:
        return "boolean"
    if features["date_ratio"] > 0.7:
        return "datetime"
    if features["numeric_ratio"] > 0.9:
        if features["unique_ratio"] < 0.05:
            return "categorical_numeric"
        return "numeric"
    if features["unique_ratio"] < 0.1:
        return "categorical"
    return "text"


# ── Training data generation ──────────────────────────────────────────────────

def generate_training_data():
    """Generate synthetic training samples for column type classifier."""
    np.random.seed(42)
    samples = []
    labels = []

    # Numeric columns
    for _ in range(200):
        s = pd.Series(np.random.randn(100))
        samples.append(extract_column_features(s))
        labels.append("numeric")

    # Categorical columns
    for _ in range(200):
        s = pd.Series(np.random.choice(["A", "B", "C", "D"], 100))
        samples.append(extract_column_features(s))
        labels.append("categorical")

    # Boolean columns
    for _ in range(150):
        s = pd.Series(np.random.choice(["true", "false"], 100))
        samples.append(extract_column_features(s))
        labels.append("boolean")

    # Datetime columns
    for _ in range(150):
        dates = pd.date_range("2020-01-01", periods=100, freq="D").astype(str)
        s = pd.Series(dates)
        samples.append(extract_column_features(s))
        labels.append("datetime")

    # Text columns
    for _ in range(150):
        words = ["hello world this is a text", "another long description here",
                 "some random text content", "user generated content example"]
        s = pd.Series(np.random.choice(words, 100))
        samples.append(extract_column_features(s))
        labels.append("text")

    # Categorical numeric columns
    for _ in range(150):
        s = pd.Series(np.random.choice([1, 2, 3, 4, 5], 100).astype(float))
        samples.append(extract_column_features(s))
        labels.append("categorical_numeric")

    return pd.DataFrame(samples), labels


# ── Train & save model ────────────────────────────────────────────────────────

def train_column_type_model():
    """Train and save the column type classifier."""
    print("Training column type classifier...")
    X, y = generate_training_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_enc)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model, le


def load_or_train_model():
    """Load model if exists, else train a new one."""
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data["model"], data["label_encoder"]
    return train_column_type_model()


# ── Data quality report ───────────────────────────────────────────────────────

def analyze_data_quality(df: pd.DataFrame) -> dict:
    """
    Full data quality report for a DataFrame.
    Returns detected types, issues, and recommendations.
    """
    model, le = load_or_train_model()
    results = {}

    for col in df.columns:
        series = df[col]
        features = extract_column_features(series)
        feature_df = pd.DataFrame([features])

        predicted_enc = model.predict(feature_df)[0]
        predicted_type = le.inverse_transform([predicted_enc])[0]

        # Quality issues
        issues = []
        null_pct = features["null_ratio"] * 100
        if null_pct > 0:
            issues.append(f"{null_pct:.1f}% missing values")
        if features["unique_ratio"] == 1.0 and predicted_type == "categorical":
            issues.append("All values unique — may be an ID column")
        if features["unique_ratio"] == 0.0:
            issues.append("All values identical — low information")
        if predicted_type == "numeric" and features["unique_ratio"] < 0.02:
            issues.append("Very low cardinality for numeric — consider treating as categorical")

        results[col] = {
            "detected_type": predicted_type,
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(null_pct, 2),
            "unique_values": int(df[col].nunique()),
            "issues": issues,
        }

    return results