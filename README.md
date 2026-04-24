# AI Powered Data Analyst

This repository contains a local AI-powered data analysis app built with Streamlit and modular Python ML utilities.

## What Was Implemented

### 1. Core analysis modules (already present, integrated into UI)
- `data_quality.py`
  - Detects column types using extracted column features.
  - Produces data quality findings (missing values, low information columns, ID-like patterns).
- `regression.py`
  - Trains Random Forest regression models for numeric targets.
  - Returns MAE, RMSE, R2, and feature importance.
- `classification.py`
  - Trains Random Forest classification models for categorical targets.
  - Returns accuracy, classification report, confusion matrix, and feature importance.
- `clustering.py`
  - Runs KMeans clustering with optional automatic `k` selection.
  - Returns silhouette score, cluster summary, PCA coordinates, elbow/silhouette traces.
- `local_chatbot.py`
  - Local rule-based Q&A chatbot over uploaded dataset.

### 2. Streamlit UI built from scratch
- `app.py` was added and expanded to provide:
  - CSV upload and dataset preview.
  - Tabbed workflow:
    - Overview
    - Data Quality
    - Regression
    - Classification
    - Clustering
    - Local Chat
  - Visual outputs (tables + charts) instead of only raw JSON.
  - Session-state result persistence across interactions.
  - Dataset signature reset logic so old results clear when a new file is uploaded.
  - Adjustable preview row count.

### 3. Robustness fixes applied
- `regression.py`
  - `predict_regression()` now safely converts non-encoded feature inputs to float with fallback to `0.0`.
- `data_quality.py`
  - Datetime parsing changed to `format="mixed"` for better compatibility and reduced parsing noise.
- `app.py`
  - Removed pandas `Styler.background_gradient(...)` usage for correlation display to avoid matplotlib dependency runtime errors.
  - Updated deprecated Streamlit `use_container_width=True` usage to `width="stretch"`.

### 4. Dependency management added
- `requirements.txt` created with:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`
  - `streamlit`

### 5. Deployment compatibility work for Vercel
- Full Streamlit runtime is intended for local execution.
- To prevent Vercel Python build/runtime failures, a lightweight deployment entrypoint was added:
  - `main.py` exports a minimal ASGI app.
  - Root route serves a simple project landing page.
  - `/health` route returns status JSON.
- `pyproject.toml` was added with Vercel app reference:
  - `main:app`

## Current Runtime Behavior

### Local (full experience)
Run the full app locally:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501
```

### Vercel (deployment-safe experience)
- Vercel serves the lightweight ASGI landing page from `main.py`.
- This confirms deployment health but does not run the full Streamlit interactive dashboard there.

## Project Structure

- `app.py` - Streamlit dashboard UI
- `data_quality.py` - Data quality/type analysis
- `regression.py` - Regression training + prediction helpers
- `classification.py` - Classification training + prediction helpers
- `clustering.py` - Clustering analysis
- `local_chatbot.py` - Local NL Q&A on dataset
- `main.py` - Minimal ASGI entrypoint for Vercel
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Project metadata + Vercel app pointer

## Scope Note

This README documents only the work completed in this repository during this implementation cycle (UI build, integration, fixes, and deployment compatibility steps).
