"""
local_chatbot.py
Answers natural language questions about CSV data using
rule-based NLP + pandas — fully offline, no API needed.
"""

import pandas as pd
import numpy as np
import re
from typing import Optional


class LocalDataChatbot:
    """
    A rule-based chatbot that understands common data questions
    and answers them by running pandas operations on the DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = [c.lower() for c in df.columns]
        self.col_map = {c.lower(): c for c in df.columns}
        self.numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    def _find_column(self, text: str) -> Optional[str]:
        """Find which column name is mentioned in the query text."""
        text_lower = text.lower()
        # Exact match first
        for col_lower, col_orig in self.col_map.items():
            if col_lower in text_lower:
                return col_orig
        return None

    def _find_number(self, text: str) -> Optional[float]:
        """Extract first number from query text."""
        nums = re.findall(r"-?\d+\.?\d*", text)
        return float(nums[0]) if nums else None

    def answer(self, query: str) -> str:
        """
        Route a natural language query to the correct pandas operation.
        Returns a plain English answer string.
        """
        q = query.lower().strip()
        df = self.df

        # ── How many rows / shape ─────────────────────────────────────────────
        if any(w in q for w in ["how many rows", "row count", "number of rows", "total rows"]):
            return f"The dataset has **{len(df):,} rows** and **{len(df.columns)} columns**."

        if any(w in q for w in ["how many columns", "number of columns", "total columns"]):
            return f"The dataset has **{len(df.columns)} columns**: {', '.join(df.columns.tolist())}."

        # ── Column names ──────────────────────────────────────────────────────
        if any(w in q for w in ["what are the columns", "list columns", "show columns", "column names"]):
            return f"The columns are: **{', '.join(df.columns.tolist())}**."

        # ── Missing values ────────────────────────────────────────────────────
        if any(w in q for w in ["missing", "null", "nan", "empty"]):
            col = self._find_column(q)
            if col:
                count = int(df[col].isnull().sum())
                pct = round(count / len(df) * 100, 1)
                return f"Column **{col}** has **{count} missing values** ({pct}% of rows)."
            else:
                nulls = df.isnull().sum()
                nulls = nulls[nulls > 0]
                if nulls.empty:
                    return "No missing values found in the dataset."
                lines = [f"- **{c}**: {v} missing ({round(v/len(df)*100,1)}%)" for c, v in nulls.items()]
                return "Missing values:\n" + "\n".join(lines)

        # ── Average / mean ────────────────────────────────────────────────────
        if any(w in q for w in ["average", "mean", "avg"]):
            col = self._find_column(q)
            if col and col in self.numeric_cols:
                val = round(df[col].mean(), 4)
                return f"The average **{col}** is **{val:,}**."
            elif not col and self.numeric_cols:
                lines = [f"- **{c}**: {round(df[c].mean(), 4):,}" for c in self.numeric_cols]
                return "Average values:\n" + "\n".join(lines)
            return f"Could not compute average — no numeric column found matching your query."

        # ── Maximum ───────────────────────────────────────────────────────────
        if any(w in q for w in ["maximum", "max", "highest", "largest", "most"]):
            col = self._find_column(q)
            if col and col in self.numeric_cols:
                val = df[col].max()
                idx = df[col].idxmax()
                return f"The maximum **{col}** is **{val:,}** (row {idx})."
            elif col and col in self.cat_cols:
                top = df[col].value_counts().idxmax()
                count = df[col].value_counts().max()
                return f"The most common **{col}** is **{top}** ({count} times)."
            elif not col and self.numeric_cols:
                lines = [f"- **{c}**: {df[c].max():,}" for c in self.numeric_cols]
                return "Maximum values:\n" + "\n".join(lines)

        # ── Minimum ───────────────────────────────────────────────────────────
        if any(w in q for w in ["minimum", "min", "lowest", "smallest", "least"]):
            col = self._find_column(q)
            if col and col in self.numeric_cols:
                val = df[col].min()
                idx = df[col].idxmin()
                return f"The minimum **{col}** is **{val:,}** (row {idx})."
            elif not col and self.numeric_cols:
                lines = [f"- **{c}**: {df[c].min():,}" for c in self.numeric_cols]
                return "Minimum values:\n" + "\n".join(lines)

        # ── Sum / total ───────────────────────────────────────────────────────
        if any(w in q for w in ["sum", "total", "add up"]):
            col = self._find_column(q)
            if col and col in self.numeric_cols:
                val = round(df[col].sum(), 4)
                return f"The total sum of **{col}** is **{val:,}**."

        # ── Unique values ─────────────────────────────────────────────────────
        if any(w in q for w in ["unique", "distinct", "different values"]):
            col = self._find_column(q)
            if col:
                n = df[col].nunique()
                if n <= 20:
                    vals = df[col].dropna().unique().tolist()
                    return f"**{col}** has **{n} unique values**: {vals}"
                return f"**{col}** has **{n} unique values**."

        # ── Value counts / frequency ──────────────────────────────────────────
        if any(w in q for w in ["count", "frequency", "how many", "distribution"]):
            col = self._find_column(q)
            if col and col in self.cat_cols:
                vc = df[col].value_counts().head(10)
                lines = [f"- **{k}**: {v}" for k, v in vc.items()]
                return f"Value counts for **{col}**:\n" + "\n".join(lines)
            elif col and col in self.numeric_cols:
                val = self._find_number(q)
                if val is not None:
                    count = int((df[col] > val).sum())
                    return f"**{count} rows** have **{col}** greater than {val}."

        # ── Correlation ───────────────────────────────────────────────────────
        if any(w in q for w in ["correlat", "relationship", "related"]):
            if len(self.numeric_cols) >= 2:
                corr = df[self.numeric_cols].corr()
                pairs = []
                cols = self.numeric_cols
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        pairs.append((cols[i], cols[j], corr.iloc[i, j]))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                top = pairs[:3]
                lines = [f"- **{a}** & **{b}**: {round(r, 3)}" for a, b, r in top]
                return "Top correlations:\n" + "\n".join(lines)
            return "Not enough numeric columns to compute correlations."

        # ── Outliers ──────────────────────────────────────────────────────────
        if any(w in q for w in ["outlier", "anomaly", "unusual", "extreme"]):
            col = self._find_column(q)
            cols_to_check = [col] if col and col in self.numeric_cols else self.numeric_cols
            results = []
            for c in cols_to_check[:5]:
                q1 = df[c].quantile(0.25)
                q3 = df[c].quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                outliers = df[(df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)]
                if len(outliers) > 0:
                    results.append(f"- **{c}**: {len(outliers)} outliers detected")
            if results:
                return "Outlier detection (IQR method):\n" + "\n".join(results)
            return "No significant outliers detected."

        # ── Median ────────────────────────────────────────────────────────────
        if "median" in q:
            col = self._find_column(q)
            if col and col in self.numeric_cols:
                val = round(df[col].median(), 4)
                return f"The median **{col}** is **{val:,}**."

        # ── Standard deviation ────────────────────────────────────────────────
        if any(w in q for w in ["std", "standard deviation", "variance", "spread"]):
            col = self._find_column(q)
            if col and col in self.numeric_cols:
                val = round(df[col].std(), 4)
                return f"The standard deviation of **{col}** is **{val:,}**."

        # ── Summary / describe ────────────────────────────────────────────────
        if any(w in q for w in ["summary", "describe", "overview", "statistics"]):
            col = self._find_column(q)
            if col and col in self.numeric_cols:
                s = df[col].describe()
                lines = [f"- {k}: {round(v, 4):,}" for k, v in s.items()]
                return f"Summary for **{col}**:\n" + "\n".join(lines)
            elif self.numeric_cols:
                lines = []
                for c in self.numeric_cols[:4]:
                    lines.append(f"**{c}** — mean: {round(df[c].mean(),2):,}, min: {df[c].min():,}, max: {df[c].max():,}")
                return "Quick summary:\n" + "\n".join(lines)

        # ── Fallback ──────────────────────────────────────────────────────────
        suggestions = [
            "average [column name]",
            "maximum [column name]",
            "how many rows",
            "missing values",
            "unique values in [column]",
            "correlation",
            "outliers in [column]",
            "summary of [column]",
        ]
        return (
            "I didn't understand that query. Try questions like:\n"
            + "\n".join(f"- *{s}*" for s in suggestions)
        )