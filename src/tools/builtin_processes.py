"""
Built-in deterministic processes (free, sample-based, no LLM/network).
Each returns structured data, never a rendered chart - chart rendering is
a UI-layer concern, kept decoupled from computation so the same result can
be rendered by Streamlit, fed to the chat agent as text, or unit-tested
without a UI at all.
"""
import pandas as pd
import numpy as np

from .registry import process, ProcessCost
from ..ingestion.data_context import DataContextObject

NUMERIC_DTYPES = {"BIGINT", "DOUBLE", "INTEGER", "FLOAT", "DECIMAL", "HUGEINT", "SMALLINT", "TINYINT", "REAL"}


def _load_sample(dco: DataContextObject) -> pd.DataFrame:
    if not dco.reservoir_sample_path:
        raise ValueError("No reservoir sample available on this DataContextObject")
    return pd.read_parquet(dco.reservoir_sample_path)


@process(
    name="missingness_report",
    description="Per-column null counts and percentages, sorted descending.",
    cost=ProcessCost.FREE,
    category="statistic",
)
def missingness_report(dco: DataContextObject, **_):
    rows = [{"column": n, "null_pct": p.null_pct, "null_count": p.null_count} for n, p in dco.columns.items()]
    rows.sort(key=lambda r: r["null_pct"], reverse=True)
    return {"type": "table", "data": rows}


@process(
    name="correlation_matrix",
    description="Pearson correlation matrix over numeric columns (sample-based).",
    cost=ProcessCost.FREE,
    category="visualization",
)
def correlation_matrix(dco: DataContextObject, **_):
    df = _load_sample(dco)
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"type": "heatmap", "columns": [], "matrix": [], "note": "fewer than 2 numeric columns"}
    corr = numeric.corr(numeric_only=True)
    return {"type": "heatmap", "columns": list(corr.columns), "matrix": corr.values.tolist()}


@process(
    name="distribution_summary",
    description="Histogram bins per numeric column, value counts per low-cardinality categorical column (sample-based).",
    cost=ProcessCost.FREE,
    category="visualization",
)
def distribution_summary(dco: DataContextObject, **_):
    df = _load_sample(dco)
    result = {}
    for col in df.columns:
        prof = dco.columns.get(col)
        if prof is None:
            continue
        if prof.dtype.upper() in NUMERIC_DTYPES:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            counts, edges = np.histogram(vals, bins=30)
            result[col] = {"kind": "histogram", "counts": counts.tolist(), "bin_edges": edges.tolist()}
        elif (prof.distinct_count or 0) <= 30:
            vc = df[col].value_counts().head(30)
            result[col] = {"kind": "bar", "labels": vc.index.astype(str).tolist(), "values": vc.values.tolist()}
    return {"type": "multi_chart", "data": result}


@process(
    name="outlier_detection",
    description="Tiered outlier detection: Isolation Forest + mutual-information scoring on numeric columns (sample-based).",
    cost=ProcessCost.FREE,
    category="statistic",
)
def outlier_detection(dco: DataContextObject, **_):
    """
    Two-stage outlier detection on the reservoir sample:
      1. Isolation Forest flags which ROWS are outliers (unsupervised,
         works without knowing what "normal" looks like ahead of time).
      2. Mutual information between each numeric COLUMN and the binary
         outlier flag ranks which columns actually drive those flags -
         this is what makes the result actionable ("amount and tenure
         explain most outliers") instead of just a count.
    Returns early with a note if there's too little numeric data, or if
    every row (or no row) was flagged - MI is undefined/meaningless then.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_selection import mutual_info_classif

    df = _load_sample(dco)
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[1] == 0 or len(numeric) < 20:
        return {"type": "table", "data": [], "note": "insufficient numeric data for outlier detection"}

    filled = numeric.fillna(numeric.median())
    iso = IsolationForest(contamination="auto", random_state=42)
    is_outlier = (iso.fit_predict(filled) == -1).astype(int)

    if is_outlier.sum() == 0 or is_outlier.sum() == len(is_outlier):
        return {"type": "table", "outlier_count": int(is_outlier.sum()), "outlier_pct": round(float(is_outlier.mean()), 4), "top_contributing_columns": []}

    mi = mutual_info_classif(filled, is_outlier, random_state=42)
    contributors = sorted(zip(filled.columns, mi), key=lambda x: x[1], reverse=True)
    return {
        "type": "table",
        "outlier_count": int(is_outlier.sum()),
        "outlier_pct": round(float(is_outlier.mean()), 4),
        "top_contributing_columns": [{"column": c, "mutual_info": round(float(m), 4)} for c, m in contributors[:10]],
    }
