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
    category="model_suggestion",
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

@process(
    name="target_correlation_rank",
    description="Ranks all numeric features by their correlation with the target column.",
    cost=ProcessCost.FREE,
    requires_target=True,
    category="statistic",
)
def target_correlation_rank(dco: DataContextObject, **_):
    target_col = dco.target.column
    df = _load_sample(dco)
    
    numeric_df = df.select_dtypes(include="number")
    
    if target_col not in numeric_df.columns:
        return {
            "type": "table", 
            "data": [], 
            "note": "Target is not numeric. Pearson correlation skipped."
        }

    corr_series = numeric_df.corr(numeric_only=True)[target_col].drop(target_col, errors='ignore')
    corr_series = corr_series.sort_values(key=abs, ascending=False)
    
    rows = [{"Feature": col, "Correlation": round(val, 4)} for col, val in corr_series.dropna().items()]
    return {"type": "table", "data": rows}


@process(
    name="feature_vs_target_distributions",
    description="Distributions of numeric features grouped by the target class.",
    cost=ProcessCost.FREE,
    requires_target=True,
    category="visualization",
)
def feature_vs_target_distributions(dco: DataContextObject, **_):
    target_col = dco.target.column
    df = _load_sample(dco)

    target_prof = dco.columns.get(target_col)
    
    # Skip plotting grouped distributions if it's a regression target with too many unique values
    if target_prof and (target_prof.distinct_count or 0) > 15:
        return {
            "type": "table",
            "data": [],
            "note": f"Target '{target_col}' has >15 unique values. Grouped distributions are skipped for continuous targets."
        }

    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols = [c for c in numeric_cols if c != target_col]

    result = {}
    for col in numeric_cols:
        vals = df[[col, target_col]].dropna()
        if len(vals) == 0:
            continue

        # Create global bins for the entire feature
        counts, edges = np.histogram(vals[col], bins=20)
        
        # Count the values per class inside those exact same bins
        class_counts = {}
        for tgt_val in vals[target_col].unique():
            subset = vals[vals[target_col] == tgt_val][col]
            class_counts[str(tgt_val)], _ = np.histogram(subset, bins=edges)

        result[col] = {
            "kind": "grouped_histogram",
            "bin_edges": edges.tolist(),
            "class_counts": {k: v.tolist() for k, v in class_counts.items()}
        }

    return {"type": "multi_chart_grouped", "data": result}

@process(
    name="data_cleaning_plan",
    description="Rule-based null and dtype handling strategy (recommends drop, median, mean, or mode based on V2 heuristics).",
    cost=ProcessCost.FREE,
    category="model_suggestion",
)
def data_cleaning_plan(dco: DataContextObject, **_):
    """
    Adapts the V2 null_tools.py logic. Uses the DataContextObject to recommend
    the exact cleaning step needed for every column with missing values.
    """
    NULL_DROP_THRESHOLD = 0.60
    
    plan = []
    for col, prof in dco.columns.items():
        if prof.null_pct == 0:
            continue
            
        strategy = ""
        reason = ""
        
        # 1. High Null Threshold
        if prof.null_pct > NULL_DROP_THRESHOLD:
            strategy = "Drop Column"
            reason = f"Missing {prof.null_pct:.1%} of data (>60% threshold)."
            
        # 2. Numeric Strategies
        elif prof.dtype.upper() in NUMERIC_DTYPES:
            skew = abs(prof.skew) if prof.skew is not None else 0
            if skew > 0.5:  # Moderate to High Skew
                strategy = "Fill Median"
                reason = f"Numeric data with skewness ({prof.skew:.2f}). Median is robust to outliers."
            else:
                strategy = "Fill Mean"
                reason = "Symmetric numeric distribution."
                
        # 3. Datetime
        elif "TIME" in prof.dtype.upper() or "DATE" in prof.dtype.upper():
            strategy = "Fill Forward (ffill)"
            reason = "Standard practice for time-series/datetime data."
            
        # 4. Categorical
        else:
            strategy = "Fill Mode"
            reason = "Categorical/Text data requires most frequent value imputation."
            
        plan.append({
            "Column": col,
            "Null %": f"{prof.null_pct:.1%}",
            "Recommended Action": strategy,
            "Statistical Reason": reason
        })
        
    if not plan:
        return {"type": "table", "data": [], "note": "No missing values found. Dataset is perfectly clean!"}
        
    return {"type": "table", "data": plan}


@process(
    name="univariate_outlier_fences",
    description="Calculates IQR fences for numeric columns and recommends capping (Winsorization) vs. dropping.",
    cost=ProcessCost.FREE,
    category="statistic",
)
def univariate_outlier_fences(dco: DataContextObject, **_):
    """
    Adapts the V2 outlier_tools.py logic. Computes exact IQR fences on the
    reservoir sample and suggests Cap vs Drop based on outlier volume.
    """
    df = _load_sample(dco)
    numeric_cols = df.select_dtypes(include="number").columns
    
    OUTLIER_DROP_MAX_PCT = 0.02 # From V2 config
    
    results = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 20:
            continue
            
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        # Count outliers in the sample
        outliers = series[(series < lower_fence) | (series > upper_fence)]
        outlier_count = len(outliers)
        outlier_pct = outlier_count / len(series)
        
        if outlier_count == 0:
            continue
            
        action = "Drop Rows" if outlier_pct <= OUTLIER_DROP_MAX_PCT else "Cap (Winsorize)"
        reason = f"Outliers make up {outlier_pct:.2%} of data. " + \
                 ("Safe to drop." if action == "Drop Rows" else "Too many to drop; capping is safer.")
        
        results.append({
            "Column": col,
            "Lower Fence": round(lower_fence, 4),
            "Upper Fence": round(upper_fence, 4),
            "Outlier Count (Sample)": outlier_count,
            "Recommended Action": action,
            "Reason": reason
        })
        
    if not results:
        return {"type": "table", "data": [], "note": "No univariate IQR outliers detected in the numeric columns."}
        
    return {"type": "table", "data": results}

import json

@process(
    name="context_aware_cleaning_plan",
    description="LLM-powered cleaning strategy. Analyzes stats (min, max, skew) and infers domain context to recommend smart null/outlier handling (e.g., 'Fraud is rare, don't drop').",
    cost=ProcessCost.LLM, # This tells the UI to pass the 'llm_fn' to this function
    category="model_suggestion",
)
def context_aware_cleaning_plan(dco: DataContextObject, llm_fn=None, **_):
    if llm_fn is None:
        return {"type": "table", "data": [], "note": "Requires a valid GROQ_API_KEY to run."}

    # 1. Gather the statistical profile for columns that might have issues
    stats_payload = {}
    for col, prof in dco.columns.items():
        # Only send columns with missing data, high skew, or numeric stats to save tokens
        if prof.null_pct > 0 or (prof.skew is not None) or "DATE" in prof.dtype.upper():
            stats_payload[col] = {
                "dtype": prof.dtype,
                "null_pct": round(prof.null_pct, 4),
                "min": prof.min_val,
                "max": prof.max_val,
                "mean": prof.mean,
                "skew": prof.skew
            }

    if not stats_payload:
        return {"type": "table", "data": [], "note": "No obvious missing values or numeric skew detected."}

    # 2. Build the Prompt for the LLM
    context_str = json.dumps(dco.external_context) if dco.external_context else "Infer domain context directly from the column names."
    
    prompt = f"""
    You are an expert Principal Data Scientist. Evaluate the following dataset columns. 
    
    Domain Context:
    {context_str}
    
    Column Statistics:
    {json.dumps(stats_payload, indent=2)}
    
    Your task is to recommend a cleaning strategy for missing values and outliers for each column. 
    CRITICAL INSTRUCTION: Do not rely purely on generic statistical rules (like IQR or mean imputation). You MUST use real-world domain logic.
    - If a column is 'age' and has negative minimums, identify it as a data entry error, not a valid outlier.
    - If a column is 'transaction_amount' or 'fraud', extreme high values are legitimate heavy-tailed events. Tell the user NOT to drop them.
    - If a sensor reading has missing values, recommend forward-filling due to equipment downtime.
    
    Respond ONLY with a valid JSON array of objects. Do not use markdown formatting or backticks. 
    Each object must have exactly these keys:
    - "Column": string
    - "Identified Issue": string (e.g., "15% Nulls, Extreme Max Value")
    - "Strategy": string (e.g., "Impute with 0", "Winsorize", "Keep Outliers")
    - "Domain Reason": string (Your real-world, context-aware justification)
    """

    # 3. Call the LLM (Using the fast Llama-3 8B model injected by the UI)
    try:
        raw_response = llm_fn(prompt).strip()
        
        # Clean up markdown if the LLM ignores instructions and wraps it in ```json
        if raw_response.startswith("```"):
            raw_response = raw_response.split("```")[1]
            if raw_response.lower().startswith("json"):
                raw_response = raw_response[4:]
                
        plan_data = json.loads(raw_response.strip())
        return {"type": "table", "data": plan_data}
        
    except Exception as e:
        return {"type": "table", "data": [], "note": f"LLM failed to generate context-aware plan: {e}"}
    
@process(
    name="box_plots",
    description="Generates box plots for all numeric columns to visualize spread and outliers.",
    cost=ProcessCost.FREE,
    category="visualization",
)
def box_plots(dco: DataContextObject, **_):
    df = _load_sample(dco)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    if not numeric_cols:
        return {"type": "table", "data": [], "note": "No numeric columns available for box plots."}
    
    # Send the raw numeric sample for the UI to render natively
    return {"type": "altair_boxplots", "data": df[numeric_cols].to_dict(orient="list")}


@process(
    name="nullity_matrix",
    description="A visual matrix mapping the exact location of missing values across the dataset.",
    cost=ProcessCost.FREE,
    category="visualization",
)
def nullity_matrix(dco: DataContextObject, **_):
    df = _load_sample(dco)
    
    if df.isna().sum().sum() == 0:
        return {"type": "text", "data": "No missing values detected in the dataset."}
    
    # Downsample for faster UI rendering if the sample is very large
    if len(df) > 1000:
        df = df.sample(1000, random_state=42).sort_index()
        
    null_matrix = df.isna().astype(int).to_dict(orient="list")
    return {"type": "null_matrix", "data": null_matrix, "rows": len(df)}