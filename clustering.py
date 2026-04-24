"""
clustering.py
Groups rows in a CSV using KMeans clustering.
Automatically finds the best number of clusters using the Elbow method.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")


def preprocess_for_clustering(df: pd.DataFrame):
    """
    Prepare DataFrame for clustering:
    - Encode categorical columns
    - Fill nulls
    - Standardize features
    Returns scaled X, feature names, scaler, encoders
    """
    df = df.copy()
    encoders = {}

    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    return X_scaled, df.columns.tolist(), scaler, encoders


def find_best_k(X_scaled: np.ndarray, max_k: int = 10) -> tuple:
    """
    Use the Elbow method + Silhouette score to find optimal k.
    Returns best_k, inertias list, silhouette scores list.
    """
    max_k = min(max_k, len(X_scaled) - 1)
    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        try:
            sil = silhouette_score(X_scaled, labels)
        except Exception:
            sil = 0.0
        silhouettes.append(sil)

    # Best k = highest silhouette score
    best_k = list(k_range)[int(np.argmax(silhouettes))]
    return best_k, list(k_range), inertias, silhouettes


def train_clustering_model(df: pd.DataFrame, n_clusters: int = None) -> dict:
    """
    Train KMeans clustering on all columns of df.
    If n_clusters is None, auto-detects best k.
    Returns cluster assignments, stats, and 2D PCA coordinates for plotting.
    """
    if len(df) < 4:
        return {"error": "Need at least 4 rows to cluster."}

    X_scaled, feature_names, scaler, encoders = preprocess_for_clustering(df)

    # Find best k if not specified
    if n_clusters is None:
        best_k, k_range, inertias, silhouettes = find_best_k(X_scaled)
        elbow_data = {
            "k_range": k_range,
            "inertias": inertias,
            "silhouettes": silhouettes,
        }
    else:
        best_k = n_clusters
        elbow_data = None

    # Train final model
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Silhouette score
    try:
        sil_score = float(silhouette_score(X_scaled, labels))
    except Exception:
        sil_score = 0.0

    # PCA to 2D for visualization
    n_components = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords_2d = pca.fit_transform(X_scaled)

    # Cluster summary
    df_result = df.copy()
    df_result["cluster"] = labels

    cluster_summary = {}
    for c in range(best_k):
        subset = df_result[df_result["cluster"] == c]
        numeric_subset = subset.select_dtypes(include="number").drop(columns=["cluster"], errors="ignore")
        cluster_summary[f"Cluster {c}"] = {
            "size": int(len(subset)),
            "pct": round(len(subset) / len(df) * 100, 1),
            "means": {col: round(float(numeric_subset[col].mean()), 3)
                      for col in numeric_subset.columns[:5]},
        }

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "clustering_model.pkl")
    joblib.dump({
        "model": km,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names,
        "pca": pca,
        "n_clusters": best_k,
    }, model_path)

    return {
        "model_type": "KMeans Clustering",
        "n_clusters": best_k,
        "silhouette_score": round(sil_score, 4),
        "cluster_labels": labels.tolist(),
        "cluster_summary": cluster_summary,
        "pca_coords": coords_2d.tolist(),
        "elbow_data": elbow_data,
        "model_path": model_path,
        "feature_names": feature_names,
    }


def predict_cluster(model_path: str, input_data: dict) -> int:
    """Predict which cluster a new data point belongs to."""
    saved = joblib.load(model_path)
    model = saved["model"]
    scaler = saved["scaler"]
    encoders = saved["encoders"]
    feature_names = saved["feature_names"]

    row = []
    for col in feature_names:
        val = input_data.get(col, 0)
        if col in encoders:
            try:
                val = encoders[col].transform([str(val)])[0]
            except ValueError:
                val = 0
        else:
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
        row.append(val)

    scaled = scaler.transform([row])
    return int(model.predict(scaled)[0])