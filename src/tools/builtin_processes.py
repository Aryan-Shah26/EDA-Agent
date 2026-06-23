"""
Built-in deterministic processes (free, sample-based, no LLM/network).
Each returns structured data, never a rendered chart - chart rendering is
a UI-layer concern, kept decoupled from computation so the same result can
be rendered by Streamlit, fed to the chat agent as text, or unit-tested
without a UI at all.
"""
import pandas as pd
import numpy as np
import streamlit as st
import json

from src.agent.llm_router import get_llm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

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


# @process(
#     name="outlier_detection",
#     description="Tiered outlier detection: Isolation Forest + mutual-information scoring on numeric columns (sample-based).",
#     cost=ProcessCost.FREE,
#     category="model_suggestion",
# )
# def outlier_detection(dco: DataContextObject, **_):
#     """
#     Two-stage outlier detection on the reservoir sample:
#       1. Isolation Forest flags which ROWS are outliers (unsupervised,
#          works without knowing what "normal" looks like ahead of time).
#       2. Mutual information between each numeric COLUMN and the binary
#          outlier flag ranks which columns actually drive those flags -
#          this is what makes the result actionable ("amount and tenure
#          explain most outliers") instead of just a count.
#     Returns early with a note if there's too little numeric data, or if
#     every row (or no row) was flagged - MI is undefined/meaningless then.
#     """
#     from sklearn.ensemble import IsolationForest
#     from sklearn.feature_selection import mutual_info_classif

#     df = _load_sample(dco)
#     numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
#     if numeric.shape[1] == 0 or len(numeric) < 20:
#         return {"type": "table", "data": [], "note": "insufficient numeric data for outlier detection"}

#     filled = numeric.fillna(numeric.median())
#     iso = IsolationForest(contamination="auto", random_state=42)
#     is_outlier = (iso.fit_predict(filled) == -1).astype(int)

#     if is_outlier.sum() == 0 or is_outlier.sum() == len(is_outlier):
#         return {"type": "table", "outlier_count": int(is_outlier.sum()), "outlier_pct": round(float(is_outlier.mean()), 4), "top_contributing_columns": []}

#     mi = mutual_info_classif(filled, is_outlier, random_state=42)
#     contributors = sorted(zip(filled.columns, mi), key=lambda x: x[1], reverse=True)
#     return {
#         "type": "table",
#         "outlier_count": int(is_outlier.sum()),
#         "outlier_pct": round(float(is_outlier.mean()), 4),
#         "top_contributing_columns": [{"column": c, "mutual_info": round(float(m), 4)} for c, m in contributors[:10]],
#     }

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


# @process(
#     name="univariate_outlier_fences",
#     description="Calculates IQR fences for numeric columns and recommends capping (Winsorization) vs. dropping.",
#     cost=ProcessCost.FREE,
#     category="statistic",
# )
# def univariate_outlier_fences(dco: DataContextObject, **_):
#     df = _load_sample(dco)
#     numeric_cols = df.select_dtypes(include="number").columns
    
#     OUTLIER_DROP_MAX_PCT = 0.02 # From V2 config
    
#     results = []
#     for col in numeric_cols:
#         series = df[col].dropna()
#         if len(series) < 20:
#             continue
            
#         q1 = series.quantile(0.25)
#         q3 = series.quantile(0.75)
#         iqr = q3 - q1
#         lower_fence = q1 - 1.5 * iqr
#         upper_fence = q3 + 1.5 * iqr
        
#         # Count outliers in the sample
#         outliers = series[(series < lower_fence) | (series > upper_fence)]
#         outlier_count = len(outliers)
#         outlier_pct = outlier_count / len(series)
        
#         if outlier_count == 0:
#             continue
            
#         action = "Drop Rows" if outlier_pct <= OUTLIER_DROP_MAX_PCT else "Cap (Winsorize)"
#         reason = f"Outliers make up {outlier_pct:.2%} of data. " + \
#                  ("Safe to drop." if action == "Drop Rows" else "Too many to drop; capping is safer.")
        
#         results.append({
#             "Column": col,
#             "Lower Fence": round(lower_fence, 4),
#             "Upper Fence": round(upper_fence, 4),
#             "Outlier Count (Sample)": outlier_count,
#             "Recommended Action": action,
#             "Reason": reason
#         })
        
#     if not results:
#         return {"type": "table", "data": [], "note": "No univariate IQR outliers detected in the numeric columns."}
        
#     return {"type": "table", "data": results}


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

@process(
    name="run_automl_suite",
    description="Trains multiple baseline algorithms on the reservoir sample and compares performance metrics.",
    cost=ProcessCost.FREE,
    category="automl"
)
def run_automl_suite(dco: DataContextObject, **_):
    df = _load_sample(dco)
    target = dco.target.column
    
    if target not in df.columns:
        return {"type": "text", "data": "Target column not found in data sample."}
        
    df = df.dropna(subset=[target])
    if len(df) < 40:
        return {"type": "text", "data": "Insufficient row count in data sample to run training suites safely."}
        
    X = df.drop(columns=[target])
    y = df[target]
    
    # --- BULLETPROOF PREPROCESSING ---
    for col in X.columns:
        # Check if the column is text/categorical
        if X[col].dtype == 'object' or X[col].dtype.name == 'category' or pd.api.types.is_string_dtype(X[col]):
            # Impute missing text with the mode
            mode_series = X[col].mode()
            fill_val = mode_series[0] if not mode_series.empty else "Missing"
            X[col] = X[col].fillna(fill_val)
            # Encode strings to integers safely
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            # Impute missing numbers with the median
            median_val = X[col].median()
            fill_val = median_val if not pd.isna(median_val) else 0
            X[col] = X[col].fillna(fill_val)
    # ---------------------------------
    
    # Scale features for stable linear model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Task Detection
    is_classification = dco.columns[target].dtype.upper() in ["VARCHAR", "TEXT", "BOOLEAN"] or dco.columns[target].distinct_count < 20
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    leaderboard = []
    best_features = []

    if is_classification:
        y_train = LabelEncoder().fit_transform(y_train.astype(str))
        y_test = LabelEncoder().fit_transform(y_test.astype(str))
        
        # Model 1: Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        leaderboard.append({"Model": "Logistic Regression", "Accuracy": accuracy_score(y_test, lr_preds), "F1-Score": f1_score(y_test, lr_preds, average='weighted')})
        
        # Model 2: Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        leaderboard.append({"Model": "Random Forest", "Accuracy": accuracy_score(y_test, rf_preds), "F1-Score": f1_score(y_test, rf_preds, average='weighted')})
        
        best_features = rf.feature_importances_
        
    else:
        # Model 1: Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_preds = ridge.predict(X_test)
        leaderboard.append({"Model": "Ridge Regression", "R² Score": r2_score(y_test, ridge_preds), "MAE": mean_absolute_error(y_test, ridge_preds)})
        
        # Model 2: Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        leaderboard.append({"Model": "Random Forest", "R² Score": r2_score(y_test, rf_preds), "MAE": mean_absolute_error(y_test, rf_preds)})
        
        best_features = rf.feature_importances_

    # Format Feature Importance
    feature_imp = pd.DataFrame({"Feature": X.columns, "Importance": best_features})
    feature_imp = feature_imp.sort_values("Importance", ascending=False).head(10).to_dict(orient="records")

    return {
        "type": "automl_results",
        "task": "Classification" if is_classification else "Regression",
        "leaderboard": leaderboard,
        "feature_importance": feature_imp
    }

@process(
    name="analyze_outliers_smart",
    description="Uses LLM reasoning combined with statistical IQR fences to determine safe outlier handling strategies.",
    cost=ProcessCost.FREE, # Update to ProcessCost.LLM if you have a cost tracking system
    category="model_suggestion"
)
def analyze_outliers_smart(dco: DataContextObject, **_):
    df = _load_sample(dco)
    protected_target = dco.target.column if dco.target and dco.target.confirmed_by_user else None
    
    stats_payload = {}
    fences = {}
    
    # 1. The Engine: Calculate hard statistics (Skip binary flags)
    for col in df.select_dtypes(include=['number', 'float', 'int']).columns:
        # Guard Clause: Skip binary/low-cardinality columns like is_weekend entirely
        if df[col].nunique() < 5:
            continue
            
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        
        if outliers > 0:
            # Fetch the RAG domain context we injected earlier
            context = getattr(dco.columns.get(col), 'description', 'No context provided.')
            
            stats_payload[col] = {
                "outliers": outliers,
                "is_target": (col == protected_target),
                "context": context
            }
            # Save fences for the UI table later
            fences[col] = {"lower": lower, "upper": upper}

    if not stats_payload:
        return {"type": "text", "data": "No significant outliers detected in continuous numeric columns."}

    # 2. The Brain: Batched LLM Prompting
    # 2. The Brain: Batched LLM Prompting with STRICT Context Prioritization
    prompt = f"""
    You are an Expert Data Scientist analyzing dataset outliers.
    Review the following columns, their outlier counts, and the specific domain context provided by the user.

    Data to analyze:
    {json.dumps(stats_payload, indent=2)}

    --- THE GOLDEN RULE ---
    The `user_provided_context` is your ultimate source of truth. If the context implies that extreme values are normal, expected, or critical to the business logic, you MUST NOT drop or cap them. 

    For EACH column, determine the safest action:
    - "Drop Rows": ONLY if the value is biologically/physically impossible or clearly a logging error.
    - "Cap (Winsorize)": If the extreme value is valid but needs smoothing to prevent model skewing.
    - "Ignore": You MUST use this if "is_target" is true, OR if the `user_provided_context` indicates extreme values are legitimate signals (e.g., fraud amounts, flash sale spikes).

    Return a valid JSON object where keys are the column names, and values are objects with "action" and "reason".
    Format:
    {{
      "column_name": {{"action": "Ignore", "reason": "Based on the user context, these high transaction amounts represent valid VIP purchases, not errors."}}
    }}
    """
    
    # 3. Execute and Parse
    try:
        fast_llm = st.session_state.get("_test_fast_llm_override") or get_llm("fast")
        response = fast_llm.invoke(prompt).content.strip()
        
        # Clean markdown formatting if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
            
        decisions = json.loads(response)
    except Exception as e:
        return {"type": "text", "data": f"Failed to generate intelligent cleaning plan: {str(e)}"}

    # 4. Construct Final UI Payload
    outlier_data = []
    for col, stats in stats_payload.items():
        decision = decisions.get(col, {"action": "Manual Review", "reason": "LLM failed to analyze."})
        
        outlier_data.append({
            "Column": col,
            "Lower Fence": f"{fences[col]['lower']:.4f}",
            "Upper Fence": f"{fences[col]['upper']:.4f}",
            "Outlier Count": stats["outliers"],
            "Recommended Action": decision["action"],
            "Reason": decision["reason"]
        })
        
    # Return using your existing render payload type for tables
    return {
        "type": "table",
        "data": outlier_data
    }