"""
app.py
Basic Streamlit UI for the AI Powered Data Analyst project.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st

from data_quality import analyze_data_quality
from regression import train_regression_model
from classification import train_classification_model
from clustering import train_clustering_model
from local_chatbot import LocalDataChatbot


st.set_page_config(page_title="AI Powered Data Analyst", page_icon=":bar_chart:", layout="wide")


@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def _render_overview(df: pd.DataFrame) -> None:
    st.markdown("### Dataset Overview")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", f"{len(df.columns):,}")
    m3.metric("Numeric", len(numeric_cols))
    m4.metric("Categorical", len(cat_cols))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Missing Values by Column")
        missing_df = (
            df.isnull().sum()
            .rename("missing")
            .reset_index()
            .rename(columns={"index": "column"})
            .sort_values("missing", ascending=False)
        )

        if int(missing_df["missing"].sum()) == 0:
            st.info("No missing values found in this dataset.")
        else:
            st.bar_chart(missing_df.set_index("column"))

    with right:
        st.markdown("#### Column Type Composition")
        dtype_df = (
            df.dtypes.astype(str)
            .value_counts()
            .rename_axis("dtype")
            .to_frame("count")
            .sort_values("count", ascending=False)
        )
        st.bar_chart(dtype_df)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("#### Numeric Distribution")
        if numeric_cols:
            selected_numeric = st.selectbox("Select numeric column", numeric_cols, key="overview_numeric")
            hist, bin_edges = np.histogram(df[selected_numeric].dropna(), bins=20)
            hist_df = pd.DataFrame(
                {
                    "bin_start": bin_edges[:-1],
                    "count": hist,
                }
            ).set_index("bin_start")
            st.bar_chart(hist_df)
        else:
            st.info("No numeric columns available.")

    with e2:
        st.markdown("#### Top Value Frequency")
        if cat_cols:
            selected_cat = st.selectbox("Select categorical column", cat_cols, key="overview_cat")
            vc = (
                df[selected_cat]
                .astype(str)
                .value_counts(dropna=False)
                .head(10)
                .rename_axis(selected_cat)
                .to_frame("count")
            )
            st.bar_chart(vc)
        else:
            st.info("No categorical columns available.")

    st.markdown("#### Correlation Matrix")
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr(numeric_only=True)
        st.dataframe(corr_df.round(2), width="stretch")
    else:
        st.info("Need at least two numeric columns to compute correlations.")


def _render_data_quality(report: dict) -> None:
    rows = []
    for col, details in report.items():
        rows.append({
            "column": col,
            "detected_type": details.get("detected_type", "unknown"),
            "null_count": details.get("null_count", 0),
            "null_pct": details.get("null_pct", 0.0),
            "unique_values": details.get("unique_values", 0),
            "issue_count": len(details.get("issues", [])),
            "issues": " | ".join(details.get("issues", [])) if details.get("issues") else "None",
        })

    report_df = pd.DataFrame(rows)
    st.dataframe(report_df, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Missing Value Percentage")
        null_chart = report_df[["column", "null_pct"]].set_index("column")
        if float(report_df["null_pct"].sum()) == 0:
            st.info("All columns have 0% missing values.")
        else:
            st.bar_chart(null_chart)
    with c2:
        st.markdown("#### Detected Type Distribution")
        type_counts = report_df["detected_type"].value_counts().rename_axis("type").to_frame("count")
        st.bar_chart(type_counts)

    st.markdown("#### Issue Count by Column")
    issue_df = report_df[["column", "issue_count"]].set_index("column")
    if int(report_df["issue_count"].sum()) == 0:
        st.success("No quality issues were flagged by the detector.")
    else:
        st.bar_chart(issue_df)


def _render_regression(result: dict) -> None:
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", result.get("mae", "-"))
    m2.metric("RMSE", result.get("rmse", "-"))
    m3.metric("R2 Score", result.get("r2_score", "-"))

    st.markdown("#### Feature Importance")
    imp = result.get("feature_importances", {})
    if imp:
        imp_df = pd.DataFrame(list(imp.items()), columns=["feature", "importance"]).head(15)
        st.bar_chart(imp_df.set_index("feature"))

    with st.expander("Regression Raw Output"):
        st.json(result, expanded=False)


def _render_classification(result: dict) -> None:
    st.metric("Accuracy", result.get("accuracy", "-"))

    st.markdown("#### Feature Importance")
    imp = result.get("feature_importances", {})
    if imp:
        imp_df = pd.DataFrame(list(imp.items()), columns=["feature", "importance"]).head(15)
        st.bar_chart(imp_df.set_index("feature"))

    st.markdown("#### Confusion Matrix")
    classes = result.get("classes", [])
    cm = result.get("confusion_matrix", [])
    if classes and cm:
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        st.dataframe(cm_df, width="stretch")

    report = result.get("classification_report", {})
    report_rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict) and "precision" in metrics:
            report_rows.append({
                "class": label,
                "precision": round(float(metrics.get("precision", 0.0)), 4),
                "recall": round(float(metrics.get("recall", 0.0)), 4),
                "f1-score": round(float(metrics.get("f1-score", 0.0)), 4),
                "support": int(metrics.get("support", 0)),
            })

    if report_rows:
        st.markdown("#### Per-Class Metrics")
        class_df = pd.DataFrame(report_rows)
        st.dataframe(class_df, width="stretch")
        st.bar_chart(class_df.set_index("class")[["precision", "recall", "f1-score"]])

    with st.expander("Classification Raw Output"):
        st.json(result, expanded=False)


def _render_clustering(result: dict) -> None:
    c1, c2 = st.columns(2)
    c1.metric("Clusters", result.get("n_clusters", "-"))
    c2.metric("Silhouette", result.get("silhouette_score", "-"))

    labels = result.get("cluster_labels", [])
    coords = result.get("pca_coords", [])

    if labels and coords:
        coords_df = pd.DataFrame(coords, columns=["pc1", "pc2"] if len(coords[0]) > 1 else ["pc1"])
        if "pc2" not in coords_df.columns:
            coords_df["pc2"] = 0.0
        coords_df["cluster"] = [str(v) for v in labels]

        st.markdown("#### Cluster Scatter (PCA)")
        st.scatter_chart(coords_df, x="pc1", y="pc2", color="cluster")

        st.markdown("#### Cluster Size Distribution")
        size_df = coords_df["cluster"].value_counts().rename_axis("cluster").to_frame("count")
        st.bar_chart(size_df)

    elbow = result.get("elbow_data")
    if elbow:
        st.markdown("#### Elbow and Silhouette")
        elbow_df = pd.DataFrame({
            "k": elbow.get("k_range", []),
            "inertia": elbow.get("inertias", []),
            "silhouette": elbow.get("silhouettes", []),
        }).set_index("k")
        st.line_chart(elbow_df)

    with st.expander("Clustering Raw Output"):
        st.json(result, expanded=False)


def main() -> None:
    st.title("AI Powered Data Analyst")
    st.caption("Upload a CSV and explore quality checks, ML models, clustering, and Q&A chat.")

    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if not uploaded:
        st.info("Upload a CSV file to begin.")
        return

    try:
        df = load_csv(uploaded)
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return

    if df.empty:
        st.warning("The uploaded CSV is empty.")
        return

    dataset_signature = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("dataset_signature") != dataset_signature:
        st.session_state["dataset_signature"] = dataset_signature
        for key in ["dq_report", "reg_result", "cls_result", "cluster_result"]:
            st.session_state.pop(key, None)

    st.subheader("Dataset Preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Missing Cells", f"{int(df.isnull().sum().sum()):,}")
    preview_rows = st.slider("Preview rows", min_value=5, max_value=min(100, len(df)), value=min(20, len(df)))
    st.dataframe(df.head(preview_rows), width="stretch")

    tabs = st.tabs([
        "Overview",
        "Data Quality",
        "Regression",
        "Classification",
        "Clustering",
        "Local Chat",
    ])

    with tabs[0]:
        _render_overview(df)

    with tabs[1]:
        st.markdown("### Data Quality Report")
        if st.button("Run Data Quality Analysis", width="stretch"):
            with st.spinner("Analyzing columns..."):
                st.session_state["dq_report"] = analyze_data_quality(df)

        if "dq_report" in st.session_state:
            _render_data_quality(st.session_state["dq_report"])

    with tabs[2]:
        st.markdown("### Regression")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for regression target.")
        else:
            target = st.selectbox("Target Column", options=numeric_cols, key="reg_target")
            if st.button("Train Regression Model", width="stretch"):
                with st.spinner("Training regression model..."):
                    st.session_state["reg_result"] = train_regression_model(df, target)

            if "reg_result" in st.session_state:
                result = st.session_state["reg_result"]
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Regression model trained.")
                    _render_regression(result)

    with tabs[3]:
        st.markdown("### Classification")
        cat_targets = [
            c for c in df.columns
            if str(df[c].dtype) in ["object", "category"] or df[c].nunique() <= 20
        ]
        if not cat_targets:
            st.warning("No suitable categorical target columns found.")
        else:
            target = st.selectbox("Target Column", options=cat_targets, key="cls_target")
            if st.button("Train Classification Model", width="stretch"):
                with st.spinner("Training classification model..."):
                    st.session_state["cls_result"] = train_classification_model(df, target)

            if "cls_result" in st.session_state:
                result = st.session_state["cls_result"]
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Classification model trained.")
                    _render_classification(result)

    with tabs[4]:
        st.markdown("### Clustering")
        st.caption("Leave k blank to auto-pick best number of clusters.")
        k_val = st.number_input("Number of clusters (optional)", min_value=2, max_value=20, value=3)
        use_auto = st.checkbox("Auto-select best k", value=True)
        if st.button("Run Clustering", width="stretch"):
            with st.spinner("Running clustering..."):
                st.session_state["cluster_result"] = train_clustering_model(
                    df, n_clusters=None if use_auto else int(k_val)
                )

        if "cluster_result" in st.session_state:
            result = st.session_state["cluster_result"]
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Clustering completed.")
                _render_clustering(result)

    with tabs[5]:
        st.markdown("### Local Data Chatbot")
        st.caption("Ask natural-language questions about the uploaded dataset.")
        question = st.text_input("Ask a question", placeholder="Example: what are missing values?")
        if st.button("Ask", width="stretch"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                bot = LocalDataChatbot(df)
                answer = bot.answer(question)
                st.markdown(answer)

    with st.expander("Raw Columns and Types"):
        types_payload = {col: str(dtype) for col, dtype in df.dtypes.items()}
        st.code(json.dumps(types_payload, indent=2), language="json")


if __name__ == "__main__":
    main()
